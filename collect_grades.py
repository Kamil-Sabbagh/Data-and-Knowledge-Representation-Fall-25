#!/usr/bin/env python3
"""
Script to collect all grades from student PRs.

Usage:
    python collect_grades.py

Output:
    grades.csv with columns: student_name, pr_number, score, attempts, status
"""

import requests
import json
import re
import csv
from datetime import datetime

# Configuration
REPO_OWNER = "Kamil-Sabbagh"
REPO_NAME = "Data-and-Knowledge-Representation-Fall-25"
GITHUB_TOKEN = None  # Optional: set to avoid rate limits

def get_pull_requests():
    """Fetch all pull requests from the repository."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls"
    params = {"state": "all", "per_page": 100}  # Get both open and closed PRs

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()

def get_pr_comments(pr_number):
    """Fetch comments for a specific PR."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{pr_number}/comments"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def extract_grade_from_comment(comment_body):
    """Extract grade information from autograder comment."""
    # Look for "Score: XX/100" pattern
    score_match = re.search(r'Score:\s*(\d+)/(\d+)', comment_body)
    if score_match:
        score = int(score_match.group(1))
        max_score = int(score_match.group(2))
        return score, max_score
    return None, None

def extract_attempts_from_labels(labels):
    """Extract attempt count from PR labels."""
    for label in labels:
        if label['name'].startswith('attempt-'):
            try:
                return int(label['name'].split('-')[1])
            except:
                pass
    return 0

def collect_grades():
    """Main function to collect all grades."""
    print("Fetching pull requests...")
    prs = get_pull_requests()

    # Filter to only submission PRs (exclude test PRs)
    submission_prs = [pr for pr in prs if 'Submission:' in pr['title'] or 'submission/' in pr['head']['ref']]

    print(f"Found {len(submission_prs)} student submissions\n")

    grades = []

    for pr in submission_prs:
        pr_number = pr['number']
        title = pr['title']
        state = pr['state']
        created_at = pr['created_at']

        # Extract student name from title or branch
        student_name = title.replace('Submission:', '').strip()
        if not student_name:
            student_name = pr['head']['ref'].replace('submission/', '')

        # Get attempt count from labels
        attempts = extract_attempts_from_labels(pr['labels'])

        # Get comments to find grade
        print(f"Processing PR #{pr_number}: {student_name}...")
        comments = get_pr_comments(pr_number)

        score = None
        max_score = None

        # Look for autograder comment (usually from github-actions bot)
        for comment in comments:
            if '🤖 Autograder Results' in comment['body'] or 'Score:' in comment['body']:
                score, max_score = extract_grade_from_comment(comment['body'])
                if score is not None:
                    break

        # Determine status
        if score is None:
            status = "No grade yet"
        elif attempts >= 3:
            status = "Exhausted"
        elif state == "closed":
            status = "Closed"
        else:
            status = "Active"

        grades.append({
            'student_name': student_name,
            'pr_number': pr_number,
            'score': score if score is not None else 'N/A',
            'max_score': max_score if max_score is not None else 100,
            'attempts': attempts,
            'status': status,
            'pr_url': pr['html_url'],
            'created_at': created_at
        })

    # Sort by student name
    grades.sort(key=lambda x: x['student_name'].lower())

    # Write to CSV
    output_file = f'grades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['student_name', 'pr_number', 'score', 'max_score', 'attempts', 'status', 'pr_url', 'created_at']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for grade in grades:
            writer.writerow(grade)

    print(f"\n✅ Grades exported to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("GRADE SUMMARY")
    print("="*60)
    print(f"{'Student':<30} {'Score':<10} {'Attempts':<10} {'Status':<15}")
    print("-"*60)

    for grade in grades:
        score_str = f"{grade['score']}/{grade['max_score']}" if grade['score'] != 'N/A' else 'N/A'
        print(f"{grade['student_name']:<30} {score_str:<10} {grade['attempts']:<10} {grade['status']:<15}")

    print("="*60)

    # Statistics
    scored_submissions = [g for g in grades if g['score'] != 'N/A']
    if scored_submissions:
        avg_score = sum(g['score'] for g in scored_submissions) / len(scored_submissions)
        print(f"\nAverage Score: {avg_score:.1f}/100")
        print(f"Total Submissions: {len(grades)}")
        print(f"Graded: {len(scored_submissions)}")
        print(f"Pending: {len(grades) - len(scored_submissions)}")

    return grades

if __name__ == '__main__':
    try:
        grades = collect_grades()
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you're hitting rate limits, set GITHUB_TOKEN in the script.")
        print("Generate token at: https://github.com/settings/tokens")
