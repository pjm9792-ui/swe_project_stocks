import os
import secrets
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from mongo_store import MongoStore


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "output" / "web_jobs"
PIPELINE_SCRIPT = BASE_DIR / "run_pipeline.py"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    store = MongoStore()
    app.config["MONGO_STORE"] = store

    def mongo_ready() -> bool:
        return store.db is not None

    def users_col():
        if store.db is None:
            raise RuntimeError("MongoDB is not configured. Add MONGODB_URI to .env first.")
        return store.db["users"]

    def jobs_col():
        if store.db is None:
            raise RuntimeError("MongoDB is not configured. Add MONGODB_URI to .env first.")
        return store.db["web_jobs"]

    def analysis_sessions_col():
        if store.db is None:
            raise RuntimeError("MongoDB is not configured. Add MONGODB_URI to .env first.")
        return store.db["analysis_sessions"]

    def current_user() -> Optional[dict]:
        user_id = session.get("user_id")
        if not user_id or not mongo_ready():
            return None
        return users_col().find_one({"user_id": user_id})

    def login_required():
        if "user_id" not in session:
            return redirect(url_for("login"))
        return None

    def mongo_or_config_page():
        if mongo_ready():
            return None
        return render_template("config_error.html")

    def get_job_or_404(job_id: str) -> dict:
        doc = jobs_col().find_one({"job_id": job_id})
        if not doc:
            abort(404)
        if doc.get("user_id") != session.get("user_id"):
            abort(403)
        return infer_job_status(doc)

    def read_text_file(path_str: str) -> str:
        if not path_str:
            return ""
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def rel_to_base(path_str: str) -> str:
        if not path_str:
            return ""
        try:
            return str(Path(path_str).resolve().relative_to(BASE_DIR.resolve()))
        except Exception:
            return ""

    def process_is_running(pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False

    def infer_job_status(job: dict) -> dict:
        run_id = str(job.get("run_id", ""))
        run_root = BASE_DIR / "output" / "agent_runs" / run_id
        combined_path = run_root / "combined_per_stock_reports.txt"
        manifest_path = run_root / "run_manifest.json"
        current_status = str(job.get("status", "unknown"))

        analysis_doc = analysis_sessions_col().find_one({"session_key": run_id}) if mongo_ready() else None
        inferred_status = current_status

        if analysis_doc and str(analysis_doc.get("status", "")) == "completed":
            inferred_status = "completed"
        elif combined_path.exists() and manifest_path.exists():
            inferred_status = "completed"
        elif current_status == "running" and not process_is_running(job.get("pid")):
            inferred_status = "completed" if combined_path.exists() else "error"

        if inferred_status != current_status:
            updates = {
                "status": inferred_status,
                "updated_at": datetime.utcnow().isoformat(),
            }
            if manifest_path.exists():
                updates["manifest_path"] = str(manifest_path)
            if combined_path.exists():
                updates["combined_reports_path"] = str(combined_path)
            if analysis_doc:
                updates["analysis_session_id"] = str(analysis_doc.get("_id"))
            jobs_col().update_one({"job_id": job["job_id"]}, {"$set": updates})
            refreshed = jobs_col().find_one({"job_id": job["job_id"]})
            if refreshed:
                return refreshed
            return {**job, **updates}

        return job

    def wait_for_job(job_id: str, proc: subprocess.Popen, run_id: str) -> None:
        returncode = proc.wait()
        status = "completed" if returncode == 0 else "error"
        jobs_col().update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": status,
                    "return_code": int(returncode),
                    "finished_at": datetime.utcnow().isoformat(),
                }
            },
        )

        analysis_doc = analysis_sessions_col().find_one({"session_key": run_id})
        if analysis_doc:
            jobs_col().update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "analysis_session_id": str(analysis_doc.get("_id")),
                    }
                },
            )

    @app.route("/")
    def index():
        if "user_id" in session:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return mongo_problem
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            if not email or not password:
                flash("Email and password are required.")
                return redirect(url_for("register"))
            if users_col().find_one({"email": email}):
                flash("That email is already registered.")
                return redirect(url_for("register"))

            user_id = secrets.token_hex(8)
            users_col().insert_one(
                {
                    "user_id": user_id,
                    "email": email,
                    "password_hash": generate_password_hash(password),
                    "created_at": datetime.utcnow().isoformat(),
                }
            )
            session["user_id"] = user_id
            flash("Account created.")
            return redirect(url_for("dashboard"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return mongo_problem
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = users_col().find_one({"email": email})
            if not user or not check_password_hash(user.get("password_hash", ""), password):
                flash("Invalid email or password.")
                return redirect(url_for("login"))
            session["user_id"] = user["user_id"]
            flash("Logged in.")
            return redirect(url_for("dashboard"))
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("Logged out.")
        return redirect(url_for("login"))

    @app.route("/dashboard")
    def dashboard():
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return mongo_problem
        maybe_redirect = login_required()
        if maybe_redirect is not None:
            return maybe_redirect
        user = current_user()
        user_jobs = list(
            jobs_col()
            .find({"user_id": session["user_id"]}, {"_id": 0})
            .sort("created_at", -1)
        )
        return render_template("dashboard.html", user=user, jobs=user_jobs)

    @app.route("/start-analysis", methods=["POST"])
    def start_analysis():
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return mongo_problem
        maybe_redirect = login_required()
        if maybe_redirect is not None:
            return maybe_redirect

        start_rank = max(1, int(request.form.get("start_rank", 1)))
        end_rank = max(start_rank, int(request.form.get("end_rank", start_rank)))
        if start_rank < 1:
            flash("Start rank must be at least 1.")
            return redirect(url_for("dashboard"))
        if end_rank < start_rank:
            flash("End rank must be greater than or equal to start rank.")
            return redirect(url_for("dashboard"))

        run_id = f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
        job_dir = OUTPUT_ROOT / run_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "pipeline.log"

        user = current_user() or {}
        command = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--start-rank",
            str(start_rank),
            "--end-rank",
            str(end_rank),
            "--run-id",
            run_id,
            "--user-id",
            str(session["user_id"]),
        ]
        if user.get("email"):
            command.extend(["--user-email", str(user["email"])])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", str((BASE_DIR / ".mplconfig").resolve()))
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
            proc = subprocess.Popen(
                [sys.executable, "-u", *command[1:]],
                cwd=BASE_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )

        job_id = secrets.token_hex(8)
        jobs_col().insert_one(
            {
                "job_id": job_id,
                "user_id": session["user_id"],
                "run_id": run_id,
                "start_rank": start_rank,
                "end_rank": end_rank,
                "status": "running",
                "pid": int(proc.pid),
                "log_path": str(log_path),
                "created_at": datetime.utcnow().isoformat(),
            }
        )

        watcher = threading.Thread(target=wait_for_job, args=(job_id, proc, run_id), daemon=True)
        watcher.start()
        return redirect(url_for("job_detail", job_id=job_id))

    @app.route("/jobs/<job_id>")
    def job_detail(job_id: str):
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return mongo_problem
        maybe_redirect = login_required()
        if maybe_redirect is not None:
            return maybe_redirect
        job = get_job_or_404(job_id)
        return render_template("job.html", job=job)

    @app.route("/jobs/<job_id>/status")
    def job_status(job_id: str):
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return jsonify({"error": "MongoDB is not configured."}), 500
        maybe_redirect = login_required()
        if maybe_redirect is not None:
            return jsonify({"redirect": url_for("login")}), 401
        job = get_job_or_404(job_id)
        return jsonify(
            {
                "job_id": job["job_id"],
                "status": job.get("status", "unknown"),
                "run_id": job.get("run_id", ""),
                "results_url": url_for("job_results", job_id=job_id) if job.get("status") == "completed" else "",
            }
        )

    @app.route("/jobs/<job_id>/logs")
    def job_logs(job_id: str):
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return jsonify({"error": "MongoDB is not configured."}), 500
        maybe_redirect = login_required()
        if maybe_redirect is not None:
            return jsonify({"redirect": url_for("login")}), 401
        job = get_job_or_404(job_id)
        text = read_text_file(job.get("log_path", ""))
        return jsonify({"text": text, "status": job.get("status", "unknown")})

    @app.route("/jobs/<job_id>/results")
    def job_results(job_id: str):
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return mongo_problem
        maybe_redirect = login_required()
        if maybe_redirect is not None:
            return maybe_redirect
        job = get_job_or_404(job_id)
        if job.get("status") != "completed":
            return redirect(url_for("job_detail", job_id=job_id))

        analysis_doc = analysis_sessions_col().find_one({"session_key": job["run_id"]}) or {}
        combined_reports_path = analysis_doc.get("combined_reports_path", "")
        combined_text = read_text_file(combined_reports_path)

        chart_artifacts = analysis_doc.get("chart_artifacts", {}) or {}
        chart_pdf_path = chart_artifacts.get("chart_pdf_path", "")
        chart_image_paths = chart_artifacts.get("chart_image_paths", []) or []

        selected_rows = analysis_doc.get("selected_rows", []) or []
        reports = []
        for row in selected_rows:
            ticker = str(row.get("Ticker", "")).upper()
            rank = int(row.get("screen_rank", -1))
            txt_path = BASE_DIR / "output" / "agent_runs" / job["run_id"] / "per_stock" / ticker / f"{ticker}.txt"
            reports.append(
                {
                    "ticker": ticker,
                    "rank": rank,
                    "text": read_text_file(str(txt_path)),
                }
            )

        return render_template(
            "results.html",
            job=job,
            analysis=analysis_doc,
            combined_text=combined_text,
            reports=reports,
            chart_pdf_path=rel_to_base(chart_pdf_path),
            chart_image_paths=[rel_to_base(path) for path in chart_image_paths if rel_to_base(path)],
        )

    @app.route("/artifacts/<path:artifact_path>")
    def artifact(artifact_path: str):
        mongo_problem = mongo_or_config_page()
        if mongo_problem is not None:
            return mongo_problem
        maybe_redirect = login_required()
        if maybe_redirect is not None:
            return maybe_redirect
        full_path = (BASE_DIR / artifact_path).resolve()
        try:
            full_path.relative_to(BASE_DIR.resolve())
        except Exception:
            abort(403)
        if not full_path.exists() or not full_path.is_file():
            abort(404)
        return send_file(full_path)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
