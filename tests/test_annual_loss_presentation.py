from test_plain_language_ui import run_js


def test_annual_fraction_and_unavailable_reason_are_readable():
    result = run_js("""({
      legacy:modelAnnualLoss(null),
      zero:modelAnnualLoss({fraction:0,target_date:'2026-12-31'}),
      missing:modelAnnualLoss({fraction:null,reason:'Need <reviewed> support',target_date:null})
    })""", extra_scripts=["model-runs.js"])
    assert result["legacy"] == ""
    assert "0.00% of baseline tree-covered area" in result["zero"]
    assert "not the probability" in result["zero"]
    assert "Unavailable" in result["missing"]
    assert "<reviewed>" not in result["missing"]


def test_annual_only_report_preview_identifies_its_supported_output():
    from test_model_file_preview import FIXTURE
    result = run_js("(()=>{" + FIXTURE + """
      report.predictions.drivers=[];
      report.predictions.annual_loss={fraction:0,units:'fraction_0_1'};
      return modelFilePreview(report,'area');
    })()""", extra_scripts=["model-runs.js"])
    assert result["targets"] == "Annual loss fraction"
