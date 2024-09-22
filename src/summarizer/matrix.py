import argparse
import json
import os
import warnings
from typing import List

import google.generativeai as genai
import numpy as np
import pandas as pd


def extract_key_elements_with_gemini(
    descriptions: List[str], n_elements: int = 20
) -> List[str]:
    """
    Extract important elements from solution descriptions using the Gemini API.

    Args:
        descriptions (List[str]): List of solution descriptions
        n_elements (int): Number of elements to extract (default: 20)

    Returns:
        List[str]: List of extracted key elements
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config={"response_mime_type": "application/json"},
    )

    prompt = f"""
    Extract the {n_elements} most important key elements from the following solution descriptions.
    These elements should represent important concepts, techniques, or approaches that highlight differences between solutions.

    Solution descriptions:
    {descriptions}

    Output format:
    Output the list of key elements in JSON format. Example: ["element1", "element2", ...]

    Notes:
    - Descriptions are ordered by competition rank. Prioritize elements from higher-ranked solutions.
    - Avoid overly general terms and choose specific, characteristic elements.
    - Prioritize elements that clearly differentiate between solutions.
    - Include technical terms and method names if possible.
    - If an element is common to almost all solutions, break it down into more detailed sub-elements.
    """

    response = model.generate_content(prompt)
    key_elements = json.loads(response.text)
    return key_elements[:n_elements]


def create_solution_matrix_with_gemini(
    df: pd.DataFrame, key_elements: List[str]
) -> pd.DataFrame:
    """
    Create a solution matrix using the Gemini API.

    Args:
        df (pd.DataFrame): Input dataframe
        key_elements (List[str]): List of key elements

    Returns:
        pd.DataFrame: Solution matrix
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        "gemini-1.5-pro",
        generation_config={"response_mime_type": "application/json"},
    )

    matrix = []
    ranks = []
    for rank, description in df[["rank", "description"]].to_numpy():
        prompt = f"""
        Determine if the following key elements are present in the solution description.
        Respond with 1 if present, 0 if not present.

        Solution description:
        {description}

        Key elements:
        {key_elements}

        Output format:
        Output an array of 0s and 1s in JSON format. Must be in the same order as the key elements. Example: [1, 0, 1, ...]
        """

        response = model.generate_content(prompt)
        element_presence = json.loads(response.text)
        if len(element_presence) != len(key_elements):
            warnings.warn(
                f"Invalid length of element presence: {len(element_presence)}"
            )
            continue
        matrix.append(element_presence)
        ranks.append(rank)

    return pd.DataFrame(
        np.array(matrix).T, index=key_elements, columns=ranks
    ).reset_index()


def main(input_csv: str, output_csv: str, n_elements: int = 20):
    df = pd.read_csv(input_csv)

    key_elements = extract_key_elements_with_gemini(
        df["description"].tolist(), n_elements
    )
    solution_matrix = create_solution_matrix_with_gemini(df, key_elements)
    solution_matrix.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--n_elements", type=int, default=30)
    args = parser.parse_args()

    input_csv = args.input
    assert input_csv.endswith(".csv"), "Input must be a CSV file"
    output_csv = input_csv.replace(".csv", "_solution_matrix.csv")

    main(input_csv, output_csv, args.n_elements)
