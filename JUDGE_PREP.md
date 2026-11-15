# Sentinel AI: judge preparation

## One-sentence explanation

Sentinel is a simulated fraud-screening tool for a small online store. It estimates an order's fraud risk and compares the assumed costs of allowing, verifying, or declining it.

## Questions to practice

**What is the input and output?**  
The input is a simulated order, including its amount and available information about the account, device, card, and prior orders. The output is an estimated risk, an action, and a description of some signals behind it. All orders and verification outcomes in the demo are simulated.

**What is the AI doing?**  
A gradient-boosted model combines patterns in the order and its history to score fraud risk. Sentinel averages five trained models and calibrates their scores. An anomaly detector can send an unusual order for verification. The costs and actions are selected by explicit rules; Gemini, when enabled, rewrites explanations and does not decide the action.

**What did you compare it with?**  
In simulation, we compared it with rules and models using transaction fields alone. On an anonymized public card dataset, the boosted model's average precision was 0.791 versus 0.754 for logistic regression. Those intervals overlap. Average precision summarizes the precision-recall curve; higher is better, but it does not say what a live merchant would save.

**What do precision and recall mean?**  
Precision asks: of the orders we flag, what fraction are fraud? Recall asks: of the fraud that occurred, what fraction did we flag? At a validation-selected threshold in one simulated test, 377 of 402 flags were fraud (93.8% precision), and 377 of 425 frauds were flagged (88.7% recall). Those are simulated results for a score-only flag, not the three-action policy.

**Why is there a gap for guest shoppers?**  
In one simulated test, 4 of 217 genuine guest orders were flagged, compared with 21 of 20,954 genuine non-guest orders. Other signals can correlate with guest status, so removing a guest indicator alone need not fix the gap. This is a small synthetic group, not evidence of a demographic disparity in a real population.

**What is the biggest limitation?**  
The strongest results use synthetic behavior and assumed costs. The public dataset cannot check the behavioral features. The model also uses fully known simulated labels to calibrate on validation, even though a real merchant would see delayed or missing reports. The original offline policy evaluation assumes some chargebacks would arrive even for blocked orders. The live replay now avoids that particular mistake. We ran the corrected replay for one store, but the earlier three-store results have not been fully recomputed.

**What would you do next?**  
Test on independent merchant data with labels and features available at decision time, verify actual outcomes of customer checks, compare false-flag rates across relevant groups, and test whether any adjustment improves the tradeoff without causing other harm.

**How were AI tools used?**  
Claude was used to build the initial project and ChatGPT was used for subsequent testing, changes, and submission materials. Describe your own contribution accurately and include any other tools you used. The competition permits AI assistance but expects the team to understand and explain the work.
