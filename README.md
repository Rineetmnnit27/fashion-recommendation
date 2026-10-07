# Fashion Recommendation System

This project is a content-based fashion recommendation system that suggests visually similar clothing items based on an input image.

## What I Did

* Used **ResNet50** to extract useful visual features from fashion images.
* Converted each image into a feature vector representing its visual characteristics.
* Used **Cosine Similarity** to compare the input image with other products.
* Used **KNN** to find the most similar products.
* Generated the **Top 5 recommendations** for a given fashion image.

## How It Works

1. Take a fashion image as input.
2. Extract its visual features using ResNet50.
3. Compare these features with the product dataset.
4. Find the closest matches using similarity.
5. Display the top 5 visually similar products.

## Tools Used

Python, TensorFlow/Keras, ResNet50, KNN, Cosine Similarity, NumPy, Pandas

## Outcome

The system can recommend visually similar fashion products based on the appearance of the input image.

