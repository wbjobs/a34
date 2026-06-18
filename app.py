import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sympy import sympify, symbols, lambdify, Matrix
from skimage.measure import marching_cubes

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

x_sym, y_sym, z_sym = symbols("x y z")


def parse_equation(eq_str):
    if "=" in eq_str:
        lhs, rhs = eq_str.split("=", 1)
        expr = sympify(lhs) - sympify(rhs)
    else:
        expr = sympify(eq_str)
    return expr


def generate_mesh(expr_str, x_range=(-5, 5), y_range=(-5, 5), z_range=(-5, 5), resolution=60):
    expr = parse_equation(expr_str)

    f = lambdify((x_sym, y_sym, z_sym), expr, modules="numpy")

    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    spacing = (
        (x_range[1] - x_range[0]) / (resolution - 1),
        (y_range[1] - y_range[0]) / (resolution - 1),
        (z_range[1] - z_range[0]) / (resolution - 1),
    )

    try:
        F = f(X, Y, Z)
    except Exception:
        F = np.zeros_like(X, dtype=float)
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    try:
                        F[i, j, k] = float(expr.subs([(x_sym, x[i]), (y_sym, y[j]), (z_sym, z[k])]))
                    except Exception:
                        F[i, j, k] = 0.0

    F = np.nan_to_num(F, nan=0.0, posinf=1e10, neginf=-1e10)

    try:
        verts, faces, normals, _ = marching_cubes(F, level=0, spacing=spacing)
    except Exception:
        return {"vertices": [], "faces": [], "normals": []}

    verts[:, 0] += x_range[0]
    verts[:, 1] += y_range[0]
    verts[:, 2] += z_range[0]

    return {
        "vertices": verts.tolist(),
        "faces": faces.tolist(),
        "normals": normals.tolist(),
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
    resolution = data.get("resolution", 60)

    try:
        expr = parse_equation(eq_str)
    except Exception as e:
        return jsonify({"error": f"方程解析失败: {str(e)}"}), 400

    try:
        mesh = generate_mesh(eq_str, x_range, y_range, z_range, resolution)
    except Exception as e:
        return jsonify({"error": f"网格生成失败: {str(e)}"}), 500

    if not mesh["vertices"]:
        return jsonify({"error": "未找到满足方程的等值面，请调整方程或范围"}), 400

    return jsonify(mesh)


@app.route("/api/probe", methods=["POST"])
def api_probe():
    data = request.get_json()
    eq_str = data.get("equation", "x**2 + y**2 + z**2 - 4")
    px = float(data.get("x", 0))
    py = float(data.get("y", 0))
    pz = float(data.get("z", 0))

    try:
        result = compute_probe(eq_str, px, py, pz)
    except Exception as e:
        return jsonify({"error": f"探测计算失败: {str(e)}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
