import streamlit as st

from helpers.labels import DEFAULT_TARGET_LABELS
from services.api import predict_audio, predict_stream


def render_stream_result(
    payload: dict,
    *,
    success_text: str,
    windows_label: str,
    detections_label: str,
    detections_header: str | None = None,
    empty_detections_text: str,
    window_predictions_header: str | None = None,
) -> None:
    detections = payload.get("detections", [])
    rows = payload.get("window_predictions", [])

    st.success(success_text)
    st.write(f"{windows_label}: {payload.get('windows_processed', len(rows))}")
    st.write(f"{detections_label}: {payload.get('detections_count', len(detections))}")

    if detections:
        if detections_header:
            st.markdown(detections_header)
        st.dataframe(detections, use_container_width=True)
    else:
        st.info(empty_detections_text)

    if window_predictions_header:
        st.markdown(window_predictions_header)
    st.dataframe(rows, use_container_width=True, height=350)


def render_offline_inference(
    *,
    api_url: str,
    audio_bytes: bytes | None,
    audio_name: str,
    mode_label: str,
    mode_options: list[str],
    predict_button_label: str,
    predict_success_text: str,
    request_failed_text: str,
    error_prefix: str,
    stream_params_header: str,
    stream_button_label: str,
    stream_success_text: str,
    windows_label: str,
    detections_label: str,
    empty_detections_text: str,
    detections_header: str | None = None,
    window_predictions_header: str | None = None,
    widget_key_prefix: str = "",
) -> None:
    if audio_bytes is None:
        return

    mode = st.radio(
        mode_label,
        mode_options,
        index=0,
        key=f"{widget_key_prefix}mode",
    )

    if mode == mode_options[0]:
        if st.button(predict_button_label, key=f"{widget_key_prefix}predict_button"):
            try:
                response = predict_audio(api_url, audio_name, audio_bytes)
                if response.ok:
                    st.success(predict_success_text)
                    st.json(response.json())
                else:
                    st.error(f"{request_failed_text}: {response.status_code}")
                    st.code(response.text)
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"{error_prefix}: {exc}")
        return

    st.markdown(stream_params_header)
    stride_sec = st.slider(
        "Stride (seconds)",
        min_value=0.10,
        max_value=0.50,
        value=0.25,
        step=0.05,
        key=f"{widget_key_prefix}stride",
    )
    refractory_sec = st.slider(
        "Refractory period (seconds)",
        min_value=0.20,
        max_value=2.00,
        value=0.80,
        step=0.10,
        key=f"{widget_key_prefix}refractory",
    )
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.99,
        value=0.55,
        step=0.01,
        key=f"{widget_key_prefix}confidence",
    )
    target_labels_raw = st.text_input(
        "Target labels (comma-separated)",
        value=DEFAULT_TARGET_LABELS,
        key=f"{widget_key_prefix}targets",
    )

    if st.button(stream_button_label, key=f"{widget_key_prefix}stream_button"):
        try:
            response = predict_stream(
                api_url,
                audio_name,
                audio_bytes,
                stride_sec=stride_sec,
                refractory_sec=refractory_sec,
                confidence_threshold=confidence_threshold,
                target_labels_raw=target_labels_raw,
            )
            if not response.ok:
                st.error(f"{request_failed_text}: {response.status_code}")
                st.code(response.text)
                return

            render_stream_result(
                response.json(),
                success_text=stream_success_text,
                windows_label=windows_label,
                detections_label=detections_label,
                detections_header=detections_header,
                empty_detections_text=empty_detections_text,
                window_predictions_header=window_predictions_header,
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"{error_prefix}: {exc}")
