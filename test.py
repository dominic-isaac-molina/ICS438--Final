# imports
import sys # for CLI
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

from search import SearchEngine

# returns the results for the search
def print_results(results, time_taken, cluster_id):
    print(f"\n[Cluster {cluster_id}] Found {len(results)} results in {time_taken:.4f}s")

    if not results:
        print("No matches found.")
    else:
        for i, res in enumerate(results):
            print(f"{i+1}. [{res['score']:.2f}] {res['title']}")


def main():
    # load the engine( things in the components folder)
    print("Loading Search Engine...")
    engine = SearchEngine()

    # type "python3 test.py "(whatever topic)" in the terminal 
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results, time_taken, cluster_id = engine.search(query)
        print_results(results, time_taken, cluster_id)
        return


if __name__ == "__main__": main()