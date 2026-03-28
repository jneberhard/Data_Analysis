# Overview

This program is an analysis of a dataset with 4,269 loan applications that were either approved or denied. This dataset includes some basic information such as number of dependents, graduated or not, self-employed or not, annual income, loan amount, how long the loan is (in years), fico score, hard assets (residential, commercial, luxury), cash, and approved or not.
* [Web Site Name](https://www.kaggle.com/datasets/rohitgrewal/loan-approval-dataset) -- Dataset link


I wanted to analyze if being self-employed made a difference and what percentage of loans go to people above median income.
I also want to be able to put in a few preliminary numbers to see what the chance was of getting a loan.

{Provide a link to your YouTube demonstration.  It should be a 4-5 minute demo of the data set, the questions and answers, the code running and a walkthrough of the code.}

[Software Demo Video](http://youtube.link.goes.here)

# Data Analysis Results

Question #1 -  Are self‑employed applicants approved at lower rates than salaried applicants?
Answer:  The percentage of self_employed approved loans is 62.23%.
The percentage of NOT self_employed approved loans is 62.20%.
Basically makes no difference if you are self-employed

Question #2 - What percentage of approved loans go to applicants with below‑median income?
The percentage of loans to below-median income is 51.05%.
Analysis, a little over half the loans were to those with below-median income. If we had a different dataset that had the loan reason we could dig deeper and see if these low income people were buying houses, cars, or what their loans are for.


Question #3 - Given a borrower’s annual income, requested loan amount, FICO score, and assets, what is the probability of getting a loan?
This is dependent on the inputs on what the result would be.


# Development Environment

The tools used to develop the software was Visual Studio Code as the development environment. I used the Python virtual environment so it can run cleanly. I used a CSV dataset containing loan application data to do the statistical analysis.

The programming language I used was Python.
The libraries I used was Pandas to load and clean the data and Numpy to build the model and log transformations.
I wrote a custom regression implementation instead of using a library so I could learn more and dig into it deeper.

# Useful Websites

{Make a list of websites that you found helpful in this project}
* [Web Site Name](https://www.kaggle.com/datasets/rohitgrewal/loan-approval-dataset)
https://www.kaggle.com/datasets/arbaaztamboli/loan-approval-dataset
* [Web Site Name](https://www.youtube.com/watch?v=DkjCaAMBGWM ) -- Learning Pandas for Data Analysis? Start Here
* [Web Site Name](https://www.youtube.com/watch?v=EXIgjIBu4EU ) -- Learn Pandas in 30 Minutes - Python Pandas Tutorial
* [Web Site Name](https://www.youtube.com/watch?v=2uvysYbKdjM) -- Complete Python Pandas Data Science Tutorial!
* [Web Site Name](https://scikit-learn.org/stable/) -- Website for machine learning in python for predictive data analysis. Did reading about this library. Did not use because I wanted to write and understand my own code instead of using something already created. But, it is good to know there are libraries out there to lean on if needed.


# Future Work

List of things that you need to fix, improve, and add in the future.
* Add in figure payment with interest... ask for interest rate and compute monthly payment
* Find more datasets, some with more information like loan purpose to see percentages for types of loans and median income.
* More analysis, such as does having a college education make a difference?
