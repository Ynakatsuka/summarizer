import re
from typing import Dict, List, Tuple

import requests
import yaml
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


def get_kaggle_solutions(kaggle_title: str) -> Tuple[List[str], List[Dict[str, int]]]:
    """
    Retrieve solutions and their ranks for a specific Kaggle competition from the YAML file.

    Args:
        kaggle_title (str): Title of the Kaggle competition. e.g., otto-recommender-system

    Returns:
        Tuple[List[str], List[Dict[str, int]]]: A tuple containing a list of solution URLs and their ranks

    Raises:
        requests.RequestException: If an error occurs during the request
    """
    yaml_url = "https://raw.githubusercontent.com/faridrashidi/kaggle-solutions/gh-pages/_data/competitions.yml"

    try:
        response = requests.get(yaml_url)
        response.raise_for_status()
        competitions = yaml.safe_load(response.text)["competitions"]

        # Find the matching competition
        for competition in competitions:
            if kaggle_title in competition["link"]:
                solutions = []
                ranks = []
                for solution in competition.get("solutions", []):
                    solutions.append(solution.get("link"))
                    ranks.append({"rank": int(solution.get("rank"))})
                return solutions, ranks

        # If no matching competition is found
        print(f"No solutions found for the competition: {kaggle_title}")
        return [], []

    except requests.RequestException as e:
        print(f"An error occurred while fetching the YAML file: {e}")
        return [], []
    except yaml.YAMLError as e:
        print(f"An error occurred while parsing the YAML file: {e}")
        return [], []
