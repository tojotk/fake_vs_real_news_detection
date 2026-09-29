import pickle
from pathlib import Path

import sklearn
import streamlit as st


st.set_page_config(page_title="Fake vs Real News Detection", page_icon="📰", layout="wide")

MODEL_PATH = Path(__file__).parent / "fake_news_model.pkl"
VECTORIZER_PATH = Path(__file__).parent / "tfidf_vectorizer.pkl"


@st.cache_resource
def load_artifacts():
	with MODEL_PATH.open("rb") as model_file:
		model = pickle.load(model_file)
	with VECTORIZER_PATH.open("rb") as vectorizer_file:
		vectorizer = pickle.load(vectorizer_file)
	return model, vectorizer


st.title("Fake vs Real News Detection")

try:
	model, vectorizer = load_artifacts()
except (OSError, pickle.PickleError, AttributeError, ImportError) as error:
	st.error(f"Could not load the model artifacts: {error}")
	st.stop()

page = st.sidebar.radio("Page", ["Home", "Predict", "About"])
class_zero_label = st.sidebar.selectbox("Class 0 represents", ["Fake", "Real"])
class_labels = {0: class_zero_label, 1: "Real" if class_zero_label == "Fake" else "Fake"}

if page == "Home":
	st.subheader("Project overview")
	st.write(
		"A digital fact-checking organization receives thousands of news articles "
		"every day. This project provides an intelligent system to assist human "
		"editors by classifying articles as fake or real and highlighting textual "
		"patterns that influence each prediction."
	)

	st.divider()
	input_column, prediction_column, explanation_column = st.columns(3)
	with input_column:
		st.markdown("#### Article")
		st.write("Editors submit article text for an initial automated assessment.")
	with prediction_column:
		st.markdown("#### Classification")
		st.write("The model returns a fake-or-real prediction and probabilities for both classes.")
	with explanation_column:
		st.markdown("#### Textual signals")
		st.write("The app surfaces words that support or oppose the predicted class.")

	st.info(
		"This tool supports editorial review. A model prediction is not proof that "
		"an article is true or false; editors should verify sources and claims."
	)

elif page == "Predict":
	st.subheader("Article assessment")
	st.write(
		"A digital fact-checking organization receives thousands of news articles "
		"every day. This system assists human editors by classifying an article "
		"and showing text patterns that influenced its prediction."
	)

	article_text = st.text_area("News article", height=250, placeholder="Paste the article text here...")
	if st.button("Analyze article", type="primary", disabled=not article_text.strip()):
		article_vector = vectorizer.transform([article_text])
		if article_vector.nnz == 0:
			st.warning("The article contains no terms recognized by the model vocabulary.")
		else:
			probabilities = model.predict_proba(article_vector)[0]
			predicted_class = int(model.classes_[probabilities.argmax()])
			probability_by_class = {
				int(class_id): float(probability)
				for class_id, probability in zip(model.classes_, probabilities)
			}
			fake_probability = probability_by_class[
				next(class_id for class_id, label in class_labels.items() if label == "Fake")
			]
			real_probability = probability_by_class[
				next(class_id for class_id, label in class_labels.items() if label == "Real")
			]
			confidence = max(probability_by_class.values())

			st.markdown(f"### Prediction: **{class_labels[predicted_class]}**")
			st.metric("Confidence", f"{confidence:.1%}")
			fake_column, real_column = st.columns(2)
			with fake_column:
				st.progress(round(fake_probability * 100), text=f"Fake: {fake_probability:.1%}")
			with real_column:
				st.progress(round(real_probability * 100), text=f"Real: {real_probability:.1%}")

			st.subheader("Words influencing the prediction")
			base_estimators = [
				calibrated.estimator
				for calibrated in model.calibrated_classifiers_
				if hasattr(calibrated.estimator, "coef_")
			]
			if base_estimators:
				feature_names = vectorizer.get_feature_names_out()
				contributions = []
				for feature_index, tfidf_value in zip(article_vector.indices, article_vector.data):
					mean_weight = sum(
						estimator.coef_[0][feature_index] for estimator in base_estimators
					) / len(base_estimators)
					margin_contribution = float(tfidf_value * mean_weight)
					prediction_contribution = (
						margin_contribution if predicted_class == int(model.classes_[1]) else -margin_contribution
					)
					contributions.append(
						{
							"Word": feature_names[feature_index],
							"Influence": round(abs(prediction_contribution), 4),
							"Effect": "Supports prediction"
							if prediction_contribution >= 0
							else "Opposes prediction",
						}
					)
				contributions.sort(key=lambda item: item["Influence"], reverse=True)
				if contributions:
					st.dataframe(contributions[:10], hide_index=True, use_container_width=True)
				else:
					st.caption("No individual terms were available to explain this result.")
				st.caption(
					"Word influence is an approximate explanation from the classifier's "
					"linear decision weights; it is not evidence that a claim is true or false."
				)
			else:
				st.info("This model does not expose linear feature weights for word-level explanations.")

	st.info("Treat this prediction as an editorial signal, not a substitute for fact-checking.")

else:
	st.subheader("About the dataset and model")

	st.markdown("#### Dataset")
	st.warning(
		"No training dataset or dataset documentation is included in the project "
		"files. Its source, size, date range, and train/test split cannot be verified here."
	)

	st.markdown("#### Model artifacts")
	model_column, vectorizer_column, labels_column = st.columns(3)
	with model_column:
		st.metric("Classifier", type(model).__name__)
		st.caption(f"Base estimator: {type(model.estimator).__name__}")
	with vectorizer_column:
		st.metric("TF-IDF vocabulary", f"{len(vectorizer.vocabulary_):,} terms")
		st.caption("Vectorizer: TfidfVectorizer")
	with labels_column:
		st.metric("Model classes", ", ".join(map(str, model.classes_)))
		st.caption(f"Class 0 = {class_labels[0]}, Class 1 = {class_labels[1]}")

	st.write(
		"The project includes a calibrated LinearSVC classifier and a separately "
		"serialized TF-IDF vectorizer. The numeric class-to-label mapping is not "
		"stored in the model, so it can be adjusted using the Class 0 control in the sidebar."
	)
	st.caption(f"Runtime scikit-learn version: {sklearn.__version__}")
	if sklearn.__version__ != "1.6.1":
		st.warning(
			f"The model artifacts were saved with scikit-learn 1.6.1 and are loaded with "
			f"{sklearn.__version__}. Cross-version pickle compatibility is not guaranteed; "
			"verify predictions or re-export the model with the runtime version before production use."
		)
