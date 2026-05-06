"""
Problem Type Identification Module
Defines ML problem category and evaluation metrics
"""


def identify_problem_type():
    """
    Identify supervised learning problem type.
    """

    return {
        "learning_type": "Supervised Learning",
        "problem_type": "Multi-Class Classification",
        "target_variable": "Risk_Level",
        "classes": [
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ],
        "evaluation_metrics": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    }