"""
GitHub Bot for analyzing commits in Pull Requests
"""

import os
import logging
import hashlib
import hmac
from flask import Flask, request, jsonify
from github import Github
from dotenv import load_dotenv
from analyzer import VulnerabilityAnalyzer

load_dotenv()

app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")
VULNERABILITY_THRESHOLD = float(os.getenv("VULNERABILITY_THRESHOLD", "0.7"))

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")

# Initialize GitHub client and analyzer
github_client = Github(GITHUB_TOKEN)
analyzer = VulnerabilityAnalyzer()


def verify_webhook_signature(payload, signature):
    """Verify GitHub webhook signature"""
    if not WEBHOOK_SECRET:
        return True  # Skip verification if no secret is set

    expected_signature = (
        "sha256="
        + hmac.new(WEBHOOK_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    )

    return hmac.compare_digest(expected_signature, signature)


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle GitHub webhook events"""
    # Verify webhook signature
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_webhook_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 401

    event_type = request.headers.get("X-GitHub-Event")
    payload = request.json

    if event_type == "pull_request":
        action = payload.get("action")
        if action in ["opened", "synchronize"]:
            handle_pull_request(payload)

    return jsonify({"status": "success"}), 200


def handle_pull_request(payload):
    """Analyze commits in a pull request"""
    try:
        pr_number = payload["pull_request"]["number"]
        repo_full_name = payload["repository"]["full_name"]

        logging.info(f"Analyzing PR #{pr_number} in {repo_full_name}")

        # Get repository and pull request
        repo = github_client.get_repo(repo_full_name)
        pr = repo.get_pull(pr_number)

        # Get commits in the PR
        commits = pr.get_commits()

        vulnerable_commits = []
        all_results = []

        for commit in commits:
            # Get commit details
            commit_sha = commit.sha
            commit_message = commit.commit.message

            # Get changed files
            files = commit.files

            for file in files:
                if file.filename.endswith(".py"):
                    # Get file content
                    try:
                        if file.patch:  # Only analyze if there are changes
                            # Combine commit message and code changes
                            analysis_text = f"{commit_message} <SEP> {file.patch}"

                            # Analyze vulnerability
                            score = analyzer.analyze(analysis_text)

                            result = {
                                "commit": commit_sha[:7],
                                "file": file.filename,
                                "score": score,
                                "status": (
                                    "vulnerable"
                                    if score >= VULNERABILITY_THRESHOLD
                                    else "safe"
                                ),
                            }

                            all_results.append(result)

                            if score >= VULNERABILITY_THRESHOLD:
                                vulnerable_commits.append(result)

                    except Exception as e:
                        logging.error(f"Error analyzing file {file.filename}: {str(e)}")

        # Post comment on PR
        if all_results:
            post_pr_comment(pr, all_results, vulnerable_commits)

    except Exception as e:
        logging.error(f"Error handling pull request: {str(e)}")


def post_pr_comment(pr, all_results, vulnerable_commits):
    """Post analysis results as a comment on the PR"""
    # Build comment message
    comment_parts = ["## 🔍 CATastrophe Vulnerability Analysis\n"]

    if vulnerable_commits:
        comment_parts.append(
            f"⚠️ **Found {len(vulnerable_commits)} potentially vulnerable changes**\n"
        )
        comment_parts.append("### Vulnerable Commits:\n")

        for vuln in vulnerable_commits:
            comment_parts.append(
                f"- **Commit {vuln['commit']}** - `{vuln['file']}` "
                f"- Score: {vuln['score']:.3f} 🔴\n"
            )
    else:
        comment_parts.append("✅ **No vulnerabilities detected in this PR**\n")

    # Add summary table
    comment_parts.append("\n### Detailed Analysis:\n")
    comment_parts.append("| Commit | File | Score | Status |\n")
    comment_parts.append("|--------|------|-------|--------|\n")

    for result in all_results:
        status_emoji = "🔴" if result["status"] == "vulnerable" else "✅"
        comment_parts.append(
            f"| {result['commit']} | {result['file']} | "
            f"{result['score']:.3f} | {status_emoji} |\n"
        )

    comment_parts.append(f"\n*Vulnerability threshold: {VULNERABILITY_THRESHOLD}*")
    comment_parts.append(
        "\n\n---\n*Analyzed by [CATastrophe](https://github.com/your-repo/catastrophe)*"
    )

    comment_body = "".join(comment_parts)

    # Post comment
    pr.create_issue_comment(comment_body)
    logging.info(f"Posted analysis comment on PR #{pr.number}")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "analyzer": analyzer.is_loaded()}), 200


if __name__ == "__main__":
    port = int(os.getenv("BOT_PORT", 8080))
    app.run(host="0.0.0.0", port=port)
