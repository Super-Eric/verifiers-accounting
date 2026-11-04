# ans-mlops-v1 results

`traces.jsonl` from `vf-run` (`-r 1`, null harness, 16k max tokens) for each policy and mode. `oracle` and
`naive` are the scripted reference policies in `environments/ans_mlops_v1/scripts/policy_server.py`.

The Anthropic key hit its usage limit during the hard-mode and Opus runs, so those files include
rollouts that stopped with `ProviderError`. Score only traces with `ok` true and
`stop_condition == "agent_completed"`. The tables in the [environment README](../../environments/ans_mlops_v1/README.md#results) do this.
