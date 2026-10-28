 Hybrid Movie Recommendation System
it combines popularity-based ranking, content-based filtering, and user-based collaborative filtering to recommend movies. It also evaluates the component models and tunes hybrid weights using validation data.

Repository: https://github.com/CapedCrusader14/explainable-recommender-system
Table of Contents
- Overview
- Features
- Recommendation Approach
- Dataset
- Evaluation Methodology
- Latest Evaluation Results
- Project Structure
- Setup
- Run the Evaluation
- Output Files
- Limitations and Future Work
- Troubleshooting
- License
Overview
Recommendation systems help users discover items that match their interests. CineMatch explores three complementary approaches and combines their scores into a single ranking.

The project focuses on ranking movies a user may like, rather than predicting an exact star rating. The current evaluation uses MovieLens 100K ratings and movie metadata.
Features
- Popularity-based movie ranking.
- Content-based recommendation using TF-IDF and cosine similarity.
- User-based collaborative filtering using ratings from similar users.
- Hybrid ranking that combines component scores.
- Validation-based search for hybrid weights.
- Chronological per-user train, validation, and test splitting.
- Ranking evaluation with Precision@10, Recall@10, NDCG@10, MAP@10, and MRR.
- CSV outputs for evaluation metrics and weight tuning.
Recommendation Approach
1. Popularity-Based Recommendation
Ranks movies using a smoothed average-rating score with a small boost based on the number of ratings. This provides a baseline and can rank broadly well-rated movies.
2. Content-Based Filtering
Represents movie metadata—such as titles, genres, directors, and actors—as TF-IDF features. Cosine similarity estimates how closely a movie matches a user's observed preferences.
- TF-IDF gives more weight to informative terms and less weight to terms common across many items.
- Cosine similarity compares the direction of feature vectors to estimate similarity.
3. User-Based Collaborative Filtering
Uses the user–movie ratings matrix to identify users with similar rating patterns. Ratings are mean-centered before calculating user similarity. Recommendations use ratings from up to 30 positively similar users, where available.
4. Hybrid Recommendation
The component scores are normalized before being combined with weights. The evaluation script searches weight combinations on validation data, selecting the combination with the best NDCG@10 and using MAP@10 and Precision@10 as tie-breakers.

Latest recorded selected weights:
Component	Weight
Popularity	0.6
Content-based	0.3
User-based collaborative filtering	0.1


These weights are specific to the current dataset, split, and tuning procedure; they are not universally optimal.
Dataset
The project uses the MovieLens 100K ratings dataset and movie metadata.

Recorded dataset summary:
- 100,000 ratings.
- 943 users.
- 1,682 movie IDs represented in the ratings file.
- 1,681 movie records in the metadata file.
Expected data files:
- data/ml-100k/u.data — MovieLens ratings.
- data/movielens_100k.csv — movie metadata, including fields such as movie_id, title, year, directors, actors, and genres.
Make sure the data files exist at these paths before running the evaluation. If your local repository uses different filenames or paths, update the corresponding paths in the source script.
Evaluation Methodology
The evaluation script uses a chronological split of each user's ratings:
- Approximately 70% for training.
- Approximately 10% for validation.
- Approximately 20% for testing.
- Users with fewer than three ratings are kept in the training portion.
Hybrid weights are selected using the validation set. The final models are then fitted on the combined training and validation data and evaluated on the held-out test set.

Ranking evaluation settings:
- A rating of 4 or higher is treated as relevant.
- Recommendation cutoff: K = 10.
- Movies already seen by a user are excluded from candidates.
- Aggregate metrics are calculated over test users with relevant held-out items.
Metrics
Metric	What it measures
Precision@10	Fraction of the top 10 recommendations that are relevant.
Recall@10	Fraction of relevant held-out movies retrieved in the top 10.
NDCG@10	Ranking quality, giving more credit to relevant movies ranked near the top.
MAP@10	Average precision across ranked recommendations, averaged over evaluated users.
MRR	Reciprocal rank of the first relevant recommendation, averaged over evaluated users.


Latest Evaluation Results
These are the latest recorded results from the project run. Re-run the evaluation script to reproduce results in your environment.
Metric	Popularity	Content-based	User-based CF	Hybrid
Precision@10	0.0545	0.0389	0.0162	0.0615
Recall@10	0.0491	0.0342	0.0163	0.0543
NDCG@10	0.0614	0.0514	0.0199	0.0772
MAP@10	0.0264	0.0228	0.0072	0.0355
MRR	0.1216	0.1302	0.0500	0.1742


In this run, the hybrid model outperformed the popularity baseline on all five reported ranking metrics. The standalone collaborative-filtering result was weaker, which may reflect limitations in the current similarity or scoring setup and rating data.

These results come from one recorded split. They do not establish statistical significance or guarantee performance on other datasets or in a deployed application.
Project Structure
High-level view of the files used by the project. Confirm exact filenames in your local checkout if the structure has changed.
explainable-recommender-system/
├── data/
│   ├── ml-100k/
│   │   └── u.data
│   ├── movielens_100k.csv
│   ├── evaluation_results.csv       # generated by evaluation
│   ├── best_hybrid_weights.csv      # generated by evaluation
│   └── hybrid_weight_search.csv     # generated by evaluation
├── src/
│   ├── recommender.py               # content-based recommender
│   └── evaluate_recommender.py      # evaluation and hybrid weight tuning
└── README.md
Requirements
- Python 3.9 or later recommended.
- Git.
- Common Python dependencies include pandas, numpy, and scikit-learn.
If the repository contains a requirements.txt, use it as the authoritative dependency list.
Setup
1. Clone the repository
git clone https://github.com/CapedCrusader14/explainable-recommender-system.git
cd explainable-recommender-system
2. Create and activate a virtual environment (recommended)
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
If PowerShell blocks activation, use Command Prompt:
.venv\Scripts\activate.bat
macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
If requirements.txt exists:
python -m pip install -r requirements.txt
Otherwise, install the common packages used by this project:
python -m pip install pandas numpy scikit-learn
If an import error identifies another package, install it and consider recording it in requirements.txt.
Run the Evaluation
Run from the repository root:
python src/evaluate_recommender.py
The script performs the split, tunes hybrid weights using validation data, evaluates the component models and hybrid model on the test set, and writes the CSV outputs listed below.

For the content-based recommender module, inspect src/recommender.py for its available functions and expected input format. The documented end-to-end evaluation entry point is the command above.
Output Files
The evaluation script writes these files under data/:
File	Purpose
evaluation_results.csv	Final test metrics for component models and hybrid model.
best_hybrid_weights.csv	Selected hybrid weights.
hybrid_weight_search.csv	Validation metrics for candidate weight combinations.


If these files are not generated, check console output, input data paths, and write permissions.
How the Hybrid Score Works
The latest selected weights can be expressed as:
Hybrid Score =
    0.6 × Popularity Score
  + 0.3 × Content-Based Score
  + 0.1 × Collaborative-Filtering Score
where the component scores are normalized before combining them. The model ranks candidate movies using this combined score. These are the weights selected in the latest recorded evaluation, not universal settings.
Limitations and Future Work
- Cold start: New users or movies may have little or no rating history.
- Data sparsity: Users rate only a subset of movies, limiting collaborative filtering.
- Evaluation coverage: The recorded results use one chronological split. Multiple temporal cutoffs and confidence intervals would strengthen the assessment.
- User-centered evaluation: A user study could measure satisfaction and perceived usefulness.
- Diversity and novelty: Future evaluation could include diversity, novelty, and catalog coverage.
- Explainability: Add explanations such as “recommended because you liked similar genres” or “popular among users with similar ratings.”
- Model improvements: Explore item-based collaborative filtering, matrix factorization, richer metadata, and alternative weight-optimization methods.
Troubleshooting
FileNotFoundError for a dataset file
- Run the script from the repository root.
- Confirm that expected files exist in data/.
- Check that paths in src/evaluate_recommender.py match your directory structure.
ModuleNotFoundError
- Activate the intended virtual environment.
- Install dependencies from requirements.txt, if present, or use python -m pip install package-name.
Results differ from the table above
- Confirm that you use the same dataset and source-code version.
- Check split logic and dependency versions.
- Treat the values above as the latest recorded run, not guaranteed output for every environment.
Contributing
Contributions are welcome. Describe the change, explain how it was tested, and include updated evaluation results when recommendation quality changes.
1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Open a pull request describing the changes.
License
No license information is specified here. Check the repository for an existing LICENSE file before reusing or redistributing the project.
