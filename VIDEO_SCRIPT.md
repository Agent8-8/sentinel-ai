# Sentinel AI demo script

Small online stores face a tradeoff: approve a fraudulent order and lose money, or reject a real customer and lose a sale. Sentinel AI is a prototype that explores that tradeoff using simulated orders.

[Show dashboard and start the replay.]

For each order, Sentinel uses information from earlier orders and a machine-learning model to estimate risk. It then compares the estimated costs of allowing, verifying, or declining the order. An anomaly detector can also send an unusual order for verification. The dashboard shows the decision and the signals behind it.

[Show an order and its explanation, then the results or what-if view.]

In a corrected replay of one simulated store, the policy cost about fifteen thousand dollars, compared with seventy-seven thousand when every order was approved. These are simulated costs based on assumptions, not real merchant savings. We corrected the replay to avoid adding chargebacks for blocked transactions.

We also tested the model on a public credit-card fraud dataset. Its average precision was about 0.79, compared with 0.75 for logistic regression. The intervals overlap, and that dataset cannot test our behavioral features or simulated verification policy.

Sentinel is a research prototype. It would need independent merchant data and careful testing of false declines before it could be used for real payments.
