from smartdesk.ui import app as ui_app


def test_ui_app_exposes_api_info_without_crashing():
    api_info = ui_app.demo.get_api_info()

    assert "named_endpoints" in api_info
