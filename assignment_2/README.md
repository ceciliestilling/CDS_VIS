SOMETHING
Assignment description

You should include a short description of the specific assignment that you are working with. If you are working with one of the assignments that I set you, include the assignment number along with the description of the assignment. For your self-assigned project, make sure to say that it is the self-assigned project.


Methods

A short paragraph outlining how you chose to solve the problem in the assignment. For example:

This problem relates to classifying colour images. In order to address this problem, we first used   a pre-trained CNN to extract features. Next, we experimented with a simple Logistic Regression classifier to establish a baseline. Finally, we added a classification layer to the pretrained CNN, etc etc.


Usage (reproducing results)

Include some instructions for how a user can engage with your code and reproduce/replicate your results. This includes installing dependencies from a requirements.txt, and how to run the script from the command line (including extra arguments, etc).


Discussion of results

A short paragraph briefly summarizing any results along with any additional reflections. For example:
	
Our initial Logistic Regression baseline had a weighted average accuracy of 65% on the extracted features. By comparison, adding a fully-connected classification layer at the end of the pretrained CNN resulted in a score of 72%. Training curves show that training and test loss continue to decrease, suggesting that more data and training time could result in higher accuracy, etc etc.
