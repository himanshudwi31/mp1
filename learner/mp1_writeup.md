1. Which strategy won, and on what dimension? (Accuracy?
Parse rate? Cost?)

- Structured Prompt strategy won considering accuracy, Judge score and overall Latency. It has provided 100% accouracy in extraction from all 10 Job snippets

2. What surprised you? Either a strategy worked better than
expected, or worse, or a specific snippet failed in a way
you didn't predict.

- I observered two surprising facts in the analysis:
    a) Judge score is better for zero_shot (3.6) as compared to cot (3.6) where the prompt instructions were more elaborate and clear. It seems cot is better for complex reasoning where multiple steps are involed while zero_shot is better for simple tasks like this analysis. It will save both cost and have better response time.

    b) Though instructions were more detailed for structured strategy which could require little bit more response time but Latency is best for this strategy, even better than zero_shot 


3. For *your* capstone domain, which strategy would you reach
for first? Justify in 2-3 sentences.

- Based on the model comparison, I would prefer structured strategy for my capstone domain because it has scored better from accuracy and Judge score prespective. Additonally, overall Latency is lowest for this strategy and provided faster response. 
These results make it a best fit for my project.


4. If you had another day, what would you try next? (Different
model? More snippets? Different prompts?)

- If I will get another day, I would try running job snippets against different models to check prompt strategies performance if higher models are used.

I would also like to check more complex job snippets against given prompt strategies to check if changes the current verdict and some other strategy stands out better.