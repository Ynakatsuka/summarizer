import re
from typing import List

import requests
from bs4 import BeautifulSoup


def get_pdf_urls_from_github(repo_url: str) -> List[str]:
    """
    Retrieve URLs of PDF files from a GitHub repository.

    Args:
        repo_url (str): URL of the GitHub repository

    Returns:
        List[str]: A list of URLs for PDF files found in the repository

    Raises:
        requests.RequestException: If an error occurs during the request
    """
    pdf_urls = []
    try:
        # Fetch the main page of the repository
        response = requests.get(repo_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Search for links to PDF files
        for link in soup.find_all("a", href=re.compile(r"\.pdf$")):
            pdf_url = f"https://github.com{link['href']}"
            pdf_urls.append(pdf_url)

        return pdf_urls

    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return []
