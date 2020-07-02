"""Make DataHub links for all Jupyter notebooks in this project.

Run:
    $ python link.py
"""

import glob

BASE_URL = "https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=http%3A%2F%2Fgithub.com%2Fandrewqcheng%2Fling188-summer2020&urlpath=tree%2Fling188-summer2020%2F{PATH}"
SLASH_ESCAPE = "%2F"
notebooks = glob.glob("**/*.ipynb", recursive=True)
for notebook in notebooks:
    link = BASE_URL.format(PATH=notebook.replace("/", SLASH_ESCAPE))
    print(notebook, link)
    print("********")

