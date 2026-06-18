import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sympy import sympify, symbols, lambdify, Matrix
from skimage.measure import marching_cubes

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

x_sym, y_sym, z_sym = symbols("x y z")

MAX_VERTS_PER_MESH = 150000
MAX_TOTAL_VOLUME_POINTS = 80 ** 3


def parse_equation(eq_str):
    if "=" in eq_str:
        lhs, rhs = eq_str.split("=", 1)
        expr = sympify(lhs) - sympify(rhs)
    else:
        expr = sympify(eq_str)
    return expr


def _lambdify_expr(expr):
    f = lambdify((x_sym, y_sym, z_sym), expr, modules="numpy")

    def safe_f(X, Y, Z):
        try:
            out = f(X, Y, Z)
            out = np.nan_to_num(np.asarray(out, dtype=float), nan=0.0, posinf=1e10, neginf=-1e10)
        except Exception:
            arr = np.zeros(X.shape, dtype=float)
            it = np.ndindex(X.shape)
            for idx in it:
                try:
                    arr[idx] = float(expr.subs([
                        (x_sym, float(X[idx])),
                        (y_sym, float(Y[idx])),
                        (z_sym, float(Z[idx])),
                    ]))
                except Exception:
                    arr[idx] = 0.0
            out = arr
        return np.asarray(out, dtype=float)

    return safe_f


def _eval_volume(expr, xs, ys, zs):
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    F = np.zeros(X.shape, dtype=float)

    f = lambdify((x_sym, y_sym, z_sym), expr, modules="numpy")
    try:
        raw = f(X, Y, Z)
        F = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=1e10, neginf=-1e10)
    except Exception:
        for i in range(len(xs)):
            for j in range(len(ys)):
                for k in range(len(zs)):
                    try:
                        F[i, j, k] = float(expr.subs([
                            (x_sym, xs[i]), (y_sym, ys[j]), (z_sym, zs[k])
                        ]))
                    except Exception:
                        F[i, j, k] = 0.0
    return F


def _run_mc(F, xs, ys, zs):
    if F.ndim != 3 or F.shape[0] < 4 or F.shape[1] < 4 or F.shape[2] < 4:
        return None
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    z0, z1 = zs[0], zs[-1]
    nx, ny, nz = F.shape
    spacing = (
        (x1 - x0) / max(1, nx - 1),
        (y1 - y0) / max(1, ny - 1),
        (z1 - z0) / max(1, nz - 1),
    )
    try:
        verts, faces, normals, _ = marching_cubes(F, level=0, spacing=spacing)
    except Exception:
        return None
    if verts is None or len(verts) == 0:
        return None
    verts[:, 0] += x0
    verts[:, 1] += y0
    verts[:, 2] += z0
    return {"vertices": verts, "faces": faces, "normals": normals}


def _estimate_curvature_from_mesh(verts, faces):
    if len(verts) < 4:
        return np.zeros(len(verts))
    n = len(verts)
    face_normals = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]],
        verts[faces[:, 2]] - verts[faces[:, 0]],
        axis=1,
    )
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    face_normals /= norms

    vertex_normal = np.zeros((n, 3))
    for fi in range(len(faces)):
        for vi in range(3):
            vertex_normal[faces[fi, vi]] += face_normals[fi]
    vn = np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    vn[vn < 1e-12] = 1.0
    vertex_normal /= vn

    curvature = np.zeros(n)
    adj_count = np.zeros(n, dtype=int)
    for fi in range(len(faces)):
        for a in range(3):
            i = faces[fi, a]
            j = faces[fi, (a + 1) % 3]
            dot = float(np.dot(vertex_normal[i], vertex_normal[j]))
            c = 1.0 - max(-1.0, min(1.0, dot))
            curvature[i] += c
            curvature[j] += c
            adj_count[i] += 1
            adj_count[j] += 1
    adj_count[adj_count == 0] = 1
    curvature = curvature / adj_count
    return curvature


def generate_mesh_adaptive(
    expr_str,
    x_range=(-5, 5),
    y_range=(-5, 5),
    z_range=(-5, 5),
    target_resolution=60,
    level="medium",
):
    expr = parse_equation(expr_str)

    if level == "low":
        base_res = max(16, min(28, target_resolution // 2))
        refine_factor = 1
    elif level == "high":
        base_res = max(24, min(40, target_resolution))
        refine_factor = 2
    else:
        base_res = max(20, min(32, target_resolution * 3 // 4))
        refine_factor = 1

    xs_coarse = np.linspace(x_range[0], x_range[1], base_res)
    ys_coarse = np.linspace(y_range[0], y_range[1], base_res)
    zs_coarse = np.linspace(z_range[0], z_range[1], base_res)

    F_coarse = _eval_volume(expr, xs_coarse, ys_coarse, zs_coarse)
    coarse_mesh = _run_mc(F_coarse, xs_coarse, ys_coarse, zs_coarse)

    if coarse_mesh is None:
        return {"vertices": [], "faces": [], "normals": [], "bbox": None}

    verts_c = coarse_mesh["vertices"]

    if len(verts_c) == 0:
        return {"vertices": [], "faces": [], "normals": [], "bbox": None}

    margin_ratio = 0.08
    span_x = x_range[1] - x_range[0]
    span_y = y_range[1] - y_range[0]
    span_z = z_range[1] - z_range[0]

    vmin = verts_c.min(axis=0)
    vmax = verts_c.max(axis=0)
    local_bbox = (
        max(x_range[0], vmin[0] - span_x * margin_ratio),
        min(x_range[1], vmax[0] + span_x * margin_ratio),
        max(y_range[0], vmin[1] - span_y * margin_ratio),
        min(y_range[1], vmax[1] + span_y * margin_ratio),
        max(z_range[0], vmin[2] - span_z * margin_ratio),
        min(z_range[1], vmax[2] + span_z * margin_ratio),
    )

    lx0, lx1, ly0, ly1, lz0, lz1 = local_bbox
    lspan_x = max(1e-6, lx1 - lx0)
    lspan_y = max(1e-6, ly1 - ly0)
    lspan_z = max(1e-6, lz1 - lz0)

    global_span = max(lspan_x, lspan_y, lspan_z)
    desired_step = global_span / max(20.0, target_resolution)
    nx = max(16, int(lspan_x / desired_step))
    ny = max(16, int(lspan_y / desired_step))
    nz = max(16, int(lspan_z / desired_step))

    total = nx * ny * nz
    if total > MAX_TOTAL_VOLUME_POINTS:
        scale = (MAX_TOTAL_VOLUME_POINTS / total) ** (1.0 / 3.0)
        nx = max(12, int(nx * scale))
        ny = max(12, int(ny * scale))
        nz = max(12, int(nz * scale))

    if refine_factor > 1 and len(verts_c) > 50:
        curvature = _estimate_curvature_from_mesh(verts_c, coarse_mesh["faces"])
        high_curv_ratio = float(np.percentile(curvature, 75)) if len(curvature) > 0 else 0.0
    else:
        high_curv_ratio = None

    xs_fine = np.linspace(lx0, lx1, nx)
    ys_fine = np.linspace(ly0, ly1, ny)
    zs_fine = np.linspace(lz0, lz1, nz)

    F_fine = _eval_volume(expr, xs_fine, ys_fine, zs_fine)

    result = _run_mc(F_fine, xs_fine, ys_fine, zs_fine)

    if result is None:
        cv = coarse_mesh
        cv["bbox"] = local_bbox
        return cv

    verts = result["vertices"]
    faces = result["faces"]
    normals = result["normals"]

    if len(verts) > MAX_VERTS_PER_MESH:
        ratio = (MAX_VERTS_PER_MESH / len(verts)) ** (1.0 / 3.0)
        step = max(2, int(1.0 / ratio))
        nx2 = max(12, nx // step)
        ny2 = max(12, ny // step)
        nz2 = max(12, nz // step)
        xs2 = np.linspace(lx0, lx1, nx2)
        ys2 = np.linspace(ly0, ly1, ny2)
        zs2 = np.linspace(lz0, lz1, nz2)
        F2 = _eval_volume(expr, xs2, ys2, zs2)
        r2 = _run_mc(F2, xs2, ys2, zs2)
        if r2 is not None:
            verts = r2["vertices"]
            faces = r2["faces"]
            normals = r2["normals"]

    return {
        "vertices": verts,
        "faces": faces,
        "normals": normals,
        "bbox": local_bbox,
        "high_curv_ratio": high_curv_ratio,
    }


def mesh_to_jsonable(m):
    bbox = None
    if m.get("bbox") is not None:
        bbox = [float(x) for x in m["bbox"]]
    hcr = None
    if m.get("high_curv_ratio") is not None:
        hcr = float(m["high_curv_ratio"])
    return {
        "vertices": m["vertices"].tolist(),
        "faces": m["faces"].tolist(),
        "normals": m["normals"].tolist(),
        "bbox": bbox,
        "high_curv_ratio": hcr,
        "vertex_count": int(len(m["vertices"])),
        "face_count": int(len(m["faces"])),
    }


def compute_probe(expr_str, px, py, pz):
    expr = parse_equation(expr_str)

    f = lambdify((x_sym, y_sym, z_sym), expr, modules="numpy")
    try:
        residual = float(f(px, py, pz))
    except Exception:
        residual = float(expr.subs([(x_sym, px), (y_sym, py), (z_sym, pz)]))

    grad_vec = Matrix([expr.diff(s) for s in (x_sym, y_sym, z_sym)])
    grad_func = lambdify((x_sym, y_sym, z_sym), grad_vec, modules="numpy")
    try:
        g = np.array(grad_func(px, py, pz), dtype=float).flatten()
    except Exception:
        g = np.array(
            [float(v.subs([(x_sym, px), (y_sym, py), (z_sym, pz)])) for v in grad_vec],
            dtype=float,
        )

    norm = np.linalg.norm(g)
    if norm > 1e-12:
        g = g / norm

    residual = float(residual) if not (np.isnan(residual) or np.isinf(residual)) else 0.0

    return {"residual": residual, "gradient": g.tolist()}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    eq_str = data.get("equation", "x**2 + y**2 + z**2 - 4")
    x_range = tuple(data.get("x_range", [-5, 5]))
    y_range = tuple(data.get("y_range", [-5, 5]))
    z_range = tuple(data.get("z_range", [-5, 5]))
    resolution = int(data.get("resolution", 60))

    try:
        parse_equation(eq_str)
    except Exception as e:
        return jsonify({"error": f"方程解析失败: {str(e)}"}), 400

    try:
        mesh = generate_mesh_adaptive(
            eq_str, x_range, y_range, z_range, resolution, level="high"
        )
    except MemoryError:
        return jsonify({"error": "内存不足，请降低分辨率或缩小范围"}), 500
    except Exception as e:
        return jsonify({"error": f"网格生成失败: {str(e)}"}), 500

    if len(mesh["vertices"]) == 0:
        return jsonify({"error": "未找到满足方程的等值面，请调整方程或范围"}), 400

    return jsonify(mesh_to_jsonable(mesh))


@app.route("/api/generate_lod", methods=["POST"])
def api_generate_lod():
    data = request.get_json()
    eq_str = data.get("equation", "x**2 + y**2 + z**2 - 4")
    x_range = tuple(data.get("x_range", [-5, 5]))
    y_range = tuple(data.get("y_range", [-5, 5]))
    z_range = tuple(data.get("z_range", [-5, 5]))
    resolution = int(data.get("resolution", 60))

    try:
        parse_equation(eq_str)
    except Exception as e:
        return jsonify({"error": f"方程解析失败: {str(e)}"}), 400

    result = {}
    try:
        for lvl in ("low", "medium", "high"):
            m = generate_mesh_adaptive(
                eq_str, x_range, y_range, z_range, resolution, level=lvl
            )
            if len(m["vertices"]) == 0:
                return jsonify({"error": f"未找到满足方程的等值面（{lvl}级别失败）"}), 400
            result[lvl] = mesh_to_jsonable(m)
    except MemoryError:
        return jsonify({"error": "内存不足，请降低分辨率或缩小范围"}), 500
    except Exception as e:
        return jsonify({"error": f"网格生成失败: {str(e)}"}), 500

    return jsonify(result)


@app.route("/api/probe", methods=["POST"])
def api_probe():
    data = request.get_json()
    eq_str = data.get("equation", "x**2 + y**2 + z**2 - 4")
    try:
        px = float(data.get("x", 0))
        py = float(data.get("y", 0))
        pz = float(data.get("z", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "坐标无效"}), 400

    try:
        result = compute_probe(eq_str, px, py, pz)
    except Exception as e:
        return jsonify({"error": f"探测计算失败: {str(e)}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
