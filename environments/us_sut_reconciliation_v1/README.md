# us-sut-reconciliation-v1

A US **state Sales & Use Tax (SUT)** analogue of `au-bas-reconciliation-v1`, built on the v1
`verifiers` stack (`import verifiers.v1 as vf`). Each task is a synthetic quarterly ledger with
injected bookkeeping faults. The model prepares the quarterly SUT return and audits every row.

## Concept mapping (GST/BAS → US SUT)

| Australian GST / BAS | US equivalent | Implemented as |
|---|---|---|
| BAS filing instrument | Quarterly state sales & use tax return | One ledger per task, random `year`/`Q1–Q4` filing period |
| G1 total sales | Gross sales | `gross_sales`: sum of included sale lines |
| GST-free / input-taxed supplies | Exempt sales | `exempt_sales`: exempt categories + certified resales |
| 1A GST on sales | Sales tax due | `sales_tax_due`: Σ round(gross × destination combined rate) |
| 1B GST on purchases | Vendor use tax | `use_tax_due`: untaxed taxable purchases × business-ZIP rate |
| Net (1A − 1B) | Net remittance | `net_remittance = sales_tax_due + use_tax_due` |
| ABN threshold ($82.50) | Resale/exemption certificate threshold | Exemption claims above **$200** need a cert ID |

The jurisdiction is an **illustrative** single-state rulebook ("State of Lone Pine": 6.25% state
rate plus a ZIP-level local rate). Exempt categories are `unprepared_grocery`, `prescription_drug`,
`medical_device` and `consulting_service`. The full rate table and rules go in every prompt, so the
task is closed-book and needs no statutory knowledge. The rates are stylized and are not current law.

## Fault taxonomy

| Code | Rule | Deterministic condition |
|---|---|---|
| `tax_on_exempt_item` | Tax collected on an exempt good/service | Exempt-category sale with recorded tax > 0 |
| `tax_miscalculated` | Tax ≠ base × jurisdiction rate | Recorded ≠ correct on a taxable sale. Generated as a state-only rate (local omitted), an internal extraction `g·r/(1+r)`, or a keying error |
| `duplicate_invoice` | Repeated invoice number | Second (later) occurrence of an `invoice_number`, excluded from the return |
| `outside_period` | Dated outside the filing quarter | Excluded from the return |
| `missing_exemption_cert` | Exempt/resale claim without a certificate | `tax_exempt_claimed` on a taxable category, gross > $200, empty cert ID. The row is treated as taxable (sales) or use-taxable (purchases) |
| `nexus_misallocation` | Origin vs. destination sourcing | Recorded tax = gross × **origin** (business ZIP) rate instead of the delivery-ZIP rate |

Ground truth comes from `ledger.reconcile()`, using `Decimal` with half-up per-line rounding.

## Schemas

Task input: the rulebook plus a CSV ledger with the columns `transaction_id, invoice_number, date,
type (sale|purchase), category, gross_amount, tax_collected_or_paid, jurisdiction_zip,
tax_exempt_claimed, customer_or_vendor_cert_id`.

Model output: a final `<answer>` block holding JSON:

```json
{"gross_sales": 128450.00, "exempt_sales": 24100.00, "taxable_sales": 104350.00,
 "sales_tax_due": 8608.88, "use_tax_due": 412.50, "net_remittance": 9021.38,
 "faults": {"txn_014": ["tax_on_exempt_item"], "txn_038": ["missing_exemption_cert"]}}
```

## Rewards

| Reward | Weight | Definition |
|---|---|---|
| `return_correct` | 0.6 | 1.0 iff `gross_sales`, `exempt_sales`, `taxable_sales`, `sales_tax_due`, `use_tax_due`, `net_remittance` all match to the cent |
| `fault_audit` | 0.4 | Micro-F1 over `(transaction_id, fault_code)` pairs (1.0 when both sets are empty) |

Metrics (unweighted): `answer_parsed`, `fields_correct` (fraction of the 6 figures), `fault_precision`, `fault_recall`.

## Config

`--env.taskset.num-tasks` (default 50), `--env.taskset.num-rows` (30), `--env.taskset.fault-rate` (0.25),
`--env.taskset.seed` (20260701; task *i* uses seed + *i*).

## Run

```bash
uv pip install -e environments/us_sut_reconciliation_v1
uv run vf-eval us-sut-reconciliation-v1 -m <model> -n 30 --env.agent.harness.id null \
  --env.agent.runtime.type subprocess --client.base-url <openai-compatible-url> --client.api-key-var <VAR>
python scripts/summarize.py outputs/<run-dir> ...
```

`--env.agent.harness.id null` makes it a plain chat task (no tools). `--env.agent.runtime.type subprocess`
runs locally instead of in a Prime sandbox. Results are in [`results/`](../../results/README.md).
