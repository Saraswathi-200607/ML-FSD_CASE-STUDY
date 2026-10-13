from django.shortcuts import render

from ml.artifacts import ModelNotAvailableError, load_artifact


def model_info(request):
    """Public page describing the model, its measured performance and limits."""
    try:
        _, meta = load_artifact()
        error = None
    except ModelNotAvailableError as exc:
        meta, error = None, str(exc)
    return render(request, "prediction/model_info.html", {"meta": meta, "error": error})
