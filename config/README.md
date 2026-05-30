# Deployments

One YAML per deployment in `deployments/`, validated against
`microclimate.config.schema.DeploymentConfig`. Every artifact is namespaced by
`deployment_id`.

To add a microclimate: copy `lethbridge.yml`, set the `deployment_id`, `target` (a station
with a registered **deep**-history connector), `neighbors`, and `output.forecast_json`, then
run training for it. CI (`tests/config/test_deployments_valid.py`) asserts the new config
loads and that every named source is registered and deep.
