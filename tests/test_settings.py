from config.settings import Settings


def test_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv("APP_NAME", "Investor Intelligence Test")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings(_env_file=None)

    assert settings.app_name == "Investor Intelligence Test"
    assert settings.debug is True
