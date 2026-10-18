# Your dashboard website (free, runs in the cloud)

Once set up, GitHub runs the daily signal every weekday at about 4:30 pm IST on its own
computers, saves the paper book, and updates your dashboard page. Your laptop can be off.
You open the page from your phone or laptop: `https://<your-github-username>.github.io/trading-agents/`

Cost: free. One-time setup: about 20 minutes.

Note: on the free plan the website needs a **public** repository, so anyone who finds the link
can see the code and the paper results. There is no money, account or personal data in them.

## Steps

1. **Create a GitHub account** at https://github.com/signup (free).
2. **Install GitHub Desktop** from https://desktop.github.com and sign in with that account.
3. In your `trading_agents` folder, double-click **`install_website_files.bat`** (puts the cloud
   schedule file in the hidden `.github\workflows` folder; it should say "Done").
4. In GitHub Desktop: **File -> Add local repository** -> choose your `trading_agents` folder.
   It will say the folder is not a repository: click **create a repository**.
   - Name: `trading-agents` -> **Create repository**.
   - (The `.gitignore` file already keeps the big price cache, logs and reports out.)
5. Click **Publish repository**. **Untick "Keep this code private"**, then **Publish**.
6. On github.com open your new repository:
   - **Settings -> Pages ->** under "Build and deployment", Source: **GitHub Actions**.
   - **Settings -> Actions -> General ->** "Workflow permissions": **Read and write permissions** -> **Save**.
7. Open the **Actions** tab -> **Daily paper signal** -> **Run workflow** -> **Run workflow**.
   Wait 3-5 minutes for the green tick.
8. Open `https://<your-github-username>.github.io/trading-agents/` and bookmark it on your phone.
9. On the laptop, double-click **`remove_daily_task.bat`** so the laptop stops running its own copy
   (two copies would write two different journals). From now on the cloud copy is the official one.
   To see the latest files on your laptop, click **Fetch origin -> Pull** in GitHub Desktop.

## If something goes wrong

- A red cross in the Actions tab: click it, open the failed step and send me a screenshot.
- Yahoo sometimes blocks cloud servers for a while. The program already retries; if it keeps
  failing for several days, we switch back to the laptop run and push the results from there.
- GitHub can start scheduled runs 10-60 minutes late. That is fine: the run uses the completed close.
