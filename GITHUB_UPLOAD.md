# Publish this ZIP using VS Code

1. Extract the ZIP, then open the `hcp-targeting-engine` folder in VS Code (**File > Open Folder**).
2. Open the integrated terminal. Check `git --version`. If you are prompted about author identity, run `git config --global user.name "Your Name"` and `git config --global user.email "your GitHub verified email"`.
3. Open **Source Control**. Click **Initialize Repository** (if it is not already a Git repository).
4. Stage files, commit with message `Add verification and temporal HCP targeting extensions`.
5. Choose **Publish to GitHub**. Review repository visibility and select **Public** only after confirming there are no private records, data or credentials.
6. In GitHub's **Actions** tab, verify **CI** and **Oldest supported dependencies**. The optional **mypy check** is manually dispatchable; run it too and fix any failures.
7. The new HTML dashboard can be opened directly from `examples/demo_dashboard.html`. Everything bundled there is synthetic.

The ZIP is the local repo package, not an automatically published GitHub repository.
