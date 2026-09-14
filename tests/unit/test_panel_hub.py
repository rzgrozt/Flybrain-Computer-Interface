from flybrain_interface.panel.app import TelemetryHub


def test_slow_browser_gets_only_latest_frame() -> None:
    hub = TelemetryHub()
    client = hub.subscribe()

    hub.publish({"sequence": 1})
    hub.publish({"sequence": 2})
    hub.publish({"sequence": 3})

    assert client.qsize() == 1
    assert client.get_nowait()["sequence"] == 3
    assert hub.browser_frames_dropped == 2
