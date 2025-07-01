# SSCED: Single-Shot Crime Event Detector with Multi-View Surveillance Image Dataset

We introduce a novel benchmark dataset, which consists of crime event-based surveillance images with well-informed annotations. The dataset has crime videos and images categorized into 12 different categories, with a separate normal dataset extracted from scenes that ended five seconds before the crime, with a duration of 10 seconds. To evaluate the dataset, we narrowed down 13 categories to 6 (assault, robbery, swoon, kidnap, burglary, and normal) and conducted experiments on a total of 2,054,013 shuffled and randomly selected frames from the videos. We used four CNN models with a single transformer model for training and validation, with a smaller sub-dataset of 8,500 frames randomly extracted from each category video, totaling 51,000 frames.

# Data
We collected surveillance dataset from AI Hub. As mentioned above, the dataset consists of 6 categories with total of 2,054,013 shuffled and randomly selected frames from the videos. There are 8,500 frames randomly extracted from each category video, totaling 51,000 frames for training and validation. We provide both video dataset and frame dataset, with customizing file and 

# - Examples of dataset

<img width="1181" alt="Screen Shot 2023-03-27 at 1 14 53 PM" src="https://user-images.githubusercontent.com/90234691/227839367-085c050a-d998-4c0c-bb9c-bcebf325f908.png">

# Result

Despite a 1:40 ratio of the train-test split, the dataset performed well on common CNN models, while leaving some challenges for the Transformer model.

<img width="957" alt="Screen Shot 2023-03-27 at 1 16 28 PM" src="https://user-images.githubusercontent.com/90234691/227839597-2744b6d9-bd2c-4755-8274-2646ec3a475e.png">
