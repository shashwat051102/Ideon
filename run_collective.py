import os
from core.Collective_ideas.crew import CollectiveCrew


def run():
    """
    Run the collective-idea crew using the simplified crew().kickoff API.
    Provide inputs including:
      - seed_id (int): Required, the idea node ID to use as the seed
      - top_k (int): Optional, how many neighbors to include (default 5)
      - autolink (bool): Optional, whether to autolink if no neighbors (default True)
      - output_dir (str): Optional, folder to save the result markdown
    """
    os.makedirs('output', exist_ok=True)

    # Example inputs: adjust seed_id to an existing idea id in your DB
    inputs = {
        'seed_id': 1,
        'top_k': 5,
        'autolink': True,
        'output_dir': 'output',
    }

    result = CollectiveCrew().crew().kickoff(inputs=inputs)

    print("\n\n=== COLLECTIVE IDEA ===\n\n")
    print(result.raw)
    print("\n\nSaved to output/collective_idea.md\n")


if __name__ == "__main__":
    run()
