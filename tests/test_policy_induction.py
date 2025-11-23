"""Tests for PolicyInduction."""
# import pytest
# import pandas as pd
# import numpy as np
# from think_reason_learn.policy_induction import PolicyInduction
# from think_reason_learn.core.llms import GoogleChoice, OpenAIChoice

# X = pd.DataFrame(
#     {
#         "founder_info": [
#             "Alex is a serial entrepreneur with two successful exits, strong "
#             "network in Silicon Valley, and expertise in AI.",
#             "Jordan graduated top of class from MIT but has no prior business "
#             "experience and limited funding.",
#             "Taylor has 10 years in finance, secured seed funding quickly, and "
#             "built a talented team.",
#             "Casey started a company right out of high school, faced multiple "
#             "failures, but persists with innovative ideas.",
#             "Morgan is a former Google engineer with patents in machine learning "
#             "and venture capital backing.",
#         ]
#     }
# )

# y = np.array(["YES", "NO", "YES", "NO", "YES"]).tolist()

# sav_dir = "policy_induction"

# @pytest.mark.asyncio
# async def test_general():
#     """Test the PolicyInduction general workflow."""
#     pi = PolicyInduction(
#         gen_llmc=[
#             GoogleChoice(model="gemini-2.5-flash"),
#             OpenAIChoice(model="gpt-4o-mini"),
#         ],
#         predict_llmc=[
#             GoogleChoice(model="gemini-2.5-flash"),
#             OpenAIChoice(model="gpt-4o-mini"),
#         ],
#     )

#     pgit = await pi.set_task(
#         task_description="Predict if a startup founder will be successful "
#         "or fail based on their background.",
#     )
#     print(pgit)

#     pi = await pi.fit(X, y, reset=True)

#     df = pd.DataFrame()
#     async for sample_index, results, final_answer, token_counter in pi.predict(X):
#         print(f"Sample {sample_index}, Predict: {final_answer}")
#         df[sample_index] = results
#     print(pi.get_memory())
#     print("predictions")
#     print(df)
#     pi.save(sav_dir)
#     return pi


# @pytest.mark.asyncio
# async def test_load_and_predict():
#     pi = PolicyInduction.load(sav_dir)
#     async for idx, policy_vector, answer, token_counter in pi.predict(X):
#         print(idx, answer)
#     return pi
