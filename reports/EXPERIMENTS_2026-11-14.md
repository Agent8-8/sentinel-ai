# Follow-up experiments

These experiments use new random seeds from the same synthetic store generator. They do not establish real-world performance, fairness, or monetary savings. Scores use complete simulated validation labels for calibration and threshold selection; these labels would not all be available in a real deployment.

## Delayed-label weights

Ten alternatives were compared on validation periods from the original three development worlds, including stronger or softer age weighting, extra weight for reported fraud, and different tree settings. The best average validation average precision came from squaring the existing weights for orders with no reported fraud: 0.9100 versus 0.9061 for the original single model. Full details: `weight_experiments.json`.

That frozen candidate was then compared with the original five-model ensemble on test periods from three new worlds (seeds 31, 43, 59). Average precision was 0.8773 versus 0.8747, but mean threshold-specific precision was 0.8795 versus 0.8867, and mean recall was 0.7973 versus 0.8034. The differences are small and mixed, so the production/demo model was not changed. See `independent_weight_check.json`.

## Guest-order friction

An exploratory check compared the original model, removing its explicit guest feature, and giving more weight to unreported guest orders. Removing the guest feature did not reduce simulated guest false flags, consistent with other features carrying related information. Weighting unreported guest orders three times as heavily reduced validation guest false flags from 30 of 456 genuine guest orders to 20 of 456; mean validation average precision fell from 0.9061 to 0.9023. Unreported orders can include fraud, so this weighting is not guaranteed safe. See `guest_experiments.json`.

The 3x candidate was frozen and compared with the original five-model ensemble on three further new synthetic worlds (seeds 61, 67, 71). Each model used its own threshold selected on its world's validation data. Across these test periods, genuine guest flags fell from 43 of 588 to 18 of 588; mean precision was 0.8349 versus 0.8826 and mean recall was 0.9272 versus 0.9246 (original versus experimental). Mean average precision was 0.9041 versus 0.9047. The gap remained: 18 of 588 genuine guest orders (3.06%) were flagged versus 145 of 62,762 genuine non-guest orders (0.23%) for the experimental model. Achieved false-positive rates differ because score distributions and ties differ; this is not a matched-rate causal estimate. See `independent_guest_check.json`.

The guest-weight candidate is experimental and is not the default. It needs independent data, checks of other groups, realistic delayed labels, and an action-aware policy evaluation before anyone could justify deploying it. Guest status is a behavioral situation, not a protected demographic category.

A follow-up static policy calculation on those same, already-inspected three test worlds compared cost-based decisions without the novelty gate or trust memory. Mean simulated cost was $13,659 for the original model and $11,855 for the guest-weighted model. Mean genuine verification rates were 0.45% and 0.52%; mean false-decline rates were both about 0.058%. This is exploratory, depends on the assumed simulator costs and verification model, and uses precomputed features that do not update chargeback history in response to blocked orders. It is not an action-aware sequential policy test. See `guest_policy_check.json`.
