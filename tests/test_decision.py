"""Rastro de decisión (app/decision.py): resúmenes en español, redacción y
acumulación por turno. El cableado dentro del despacho v2 se prueba en
tests/test_stateless.py (sección "rastro de decisión")."""
from __future__ import annotations

import pytest

from app import decision
from app.decision import MAX_STEPS, MAX_SUMMARY_CHARS, DecisionTrace, summarize_step
from app.llm import LlmUsage

# ------------------------------------------------------------ resúmenes ---


def test_update_ficha_lista_nombres_de_campo_nunca_valores():
    args = {
        "rubro": "dentista",
        "dolor_principal": "no me llegan citas",
        "notas": "llamar al 0414-1234567 o a ana@correo.com",
        "tamano_aprox": None,  # None no cuenta, igual que en ToolRuntime._update_ficha
    }
    summary, ok = summarize_step("update_ficha", args, {"ok": True})
    assert ok is True
    assert summary == "actualizó lead: rubro, dolor principal, notas"
    for valor in ("dentista", "no me llegan citas", "0414", "ana@correo.com"):
        assert valor not in summary


def test_update_ficha_descarta_claves_inventadas_que_podrian_cargar_datos():
    """El LLM puede inventar claves: una que parece un teléfono o un id nunca
    se imprime — se resume como "otros campos"."""
    args = {"rubro": "spa", "58414123456": "x", "ok key!": "y"}
    summary, _ = summarize_step("update_ficha", args, {"ok": True})
    assert "58414123456" not in summary
    assert "ok key" not in summary
    assert summary == "actualizó lead: rubro, otros campos"


def test_update_ficha_sin_campos_nuevos():
    summary, ok = summarize_step(
        "update_ficha", {}, {"ok": True, "nota": "sin campos nuevos"}
    )
    assert (summary, ok) == ("actualizó lead: sin campos nuevos", True)


def test_update_ficha_fallida_lleva_razon_corta_sin_stack():
    summary, ok = summarize_step(
        "update_ficha",
        {"rubro": "spa"},
        {"ok": False, "error": "crm_error", "detalle": "Traceback (most recent call last) ..."},
    )
    assert ok is False
    assert summary == "intentó actualizar lead: rubro (falló: el CRM no respondió)"
    assert "Traceback" not in summary


def test_propose_slots_cuenta_y_lista_las_etiquetas():
    result = {
        "ok": True,
        "slots": [
            {"start_utc": "2026-10-08T14:00:00Z", "label": "jueves 8 de octubre, 10:00"},
            {"start_utc": "2026-10-09T15:00:00Z", "label": "viernes 9 de octubre, 11:00"},
            {"start_utc": "2026-10-12T14:00:00Z", "label": "lunes 12 de octubre, 10:00"},
        ],
    }
    summary, ok = summarize_step("propose_slots", {}, result)
    assert ok is True
    assert summary.startswith("ofreció 3 horarios: jueves 8 de octubre, 10:00")
    assert "lunes 12 de octubre, 10:00" in summary


def test_propose_slots_un_solo_horario_va_en_singular():
    result = {"ok": True, "slots": [{"start_utc": "x", "label": "jueves 10:00"}]}
    summary, _ = summarize_step("propose_slots", {}, result)
    assert summary == "ofreció 1 horario: jueves 10:00"


def test_propose_slots_sin_agenda_es_un_paso_fallido():
    summary, ok = summarize_step(
        "propose_slots", {}, {"ok": False, "error": "sin_agenda", "detalle": "..."}
    )
    assert ok is False
    assert summary == "buscó horarios (falló: este negocio no tiene agenda)"


def test_book_session_ok_dice_la_etiqueta_pero_no_el_enlace_de_la_reunion():
    result = {
        "ok": True,
        "label": "jueves 10:00",
        "meeting_url": "https://meet.example.com/abc-defg-hij",
        "movida": False,
    }
    summary, ok = summarize_step(
        "book_session", {"start_utc": "2026-10-08T14:00:00Z"}, result
    )
    assert (summary, ok) == ("agendó: jueves 10:00", True)
    assert "meet.example.com" not in summary


def test_reschedule_session_ok_y_fallido():
    ok_summary, ok = summarize_step(
        "reschedule_session", {}, {"ok": True, "label": "viernes 11:00", "movida": True}
    )
    assert (ok_summary, ok) == ("movió la cita a: viernes 11:00", True)
    fail_summary, ok = summarize_step(
        "reschedule_session", {}, {"ok": False, "error": "slot_taken"}
    )
    assert (fail_summary, ok) == (
        "intentó mover la cita (falló: el horario ya estaba ocupado)",
        False,
    )


def test_book_session_rechazado_por_no_ofrecido():
    summary, ok = summarize_step(
        "book_session",
        {"start_utc": "2026-10-08T14:00:00Z"},
        {"ok": False, "error": "slot_no_ofrecido", "slots_ofrecidos": []},
    )
    assert (summary, ok) == ("intentó agendar (falló: ese horario no se había ofrecido)", False)


def test_route_out_con_y_sin_recursos():
    plain, _ = summarize_step("route_out", {}, {"ok": True})
    assert plain == "marcó al lead como no calificado"
    with_res, _ = summarize_step(
        "route_out", {}, {"ok": True, "recursos": [{"label": "Guía", "url": "https://x.test"}]}
    )
    assert with_res == "marcó al lead como no calificado, compartió recursos alternativos"
    assert "x.test" not in with_res


def test_handoff_incluye_el_motivo_pero_lo_limpia():
    summary, ok = summarize_step("handoff", {"reason": "pidió humano"}, {"ok": True})
    assert (summary, ok) == ("pasó a una persona: pidió humano", True)

    sucio = {"reason": "pidió hablar al +58 414 123 4567 o ana@correo.com, ver https://wa.me/584141234567"}
    summary, _ = summarize_step("handoff", sucio, {"ok": True})
    for fuga in ("414 123", "4567", "ana@correo.com", "wa.me"):
        assert fuga not in summary
    assert summary.startswith("pasó a una persona: pidió hablar al")


def test_handoff_sin_motivo():
    summary, _ = summarize_step("handoff", {}, {"ok": True})
    assert summary == "pasó a una persona"


def test_herramienta_desconocida_es_generica_y_fallida():
    summary, ok = summarize_step(
        "buscar_conocimiento", {"query": "precios"}, {"ok": False, "error": "herramienta desconocida: buscar_conocimiento"}
    )
    assert (summary, ok) == ("usó buscar_conocimiento (falló: herramienta desconocida)", False)


def test_razon_desconocida_no_filtra_texto_libre_del_error():
    summary, _ = summarize_step(
        "propose_slots", {}, {"ok": False, "error": "Traceback: KeyError 'sk-abc12345678901234567890123456789'"}
    )
    assert summary == "buscó horarios (falló: error)"


def test_el_resumen_nunca_pasa_de_200_caracteres():
    largo = "x" * 400
    summary, _ = summarize_step("handoff", {"reason": largo}, {"ok": True})
    assert len(summary) <= MAX_SUMMARY_CHARS
    result = {"ok": True, "slots": [{"start_utc": "x", "label": f"horario número {i} muy descriptivo"} for i in range(30)]}
    summary, _ = summarize_step("propose_slots", {}, result)
    assert len(summary) <= MAX_SUMMARY_CHARS


def test_result_que_no_es_dict_cuenta_como_fallido_sin_reventar():
    summary, ok = summarize_step("route_out", {}, None)
    assert ok is False
    assert summary.startswith("intentó marcar al lead como no calificado")


# ------------------------------------------------------------ redacción ---


@pytest.mark.parametrize(
    "texto, no_debe_contener",
    [
        ("llámalo al +52 55 5000 1111 ya", "5000 1111"),
        ("0414-1234567", "1234567"),
        ("mándale a juan.perez@correo.com", "juan.perez"),
        ("entra a https://zoom.us/j/123456789?pwd=abc", "zoom.us"),
        ("usa sk-proj-abcdefghijklmnop1234", "abcdefghijklmnop"),
        ("token 0123456789abcdef0123456789abcdef0123", "0123456789abcdef"),
    ],
)
def test_clean_tacha_telefonos_correos_enlaces_y_claves(texto, no_debe_contener):
    assert no_debe_contener not in decision._clean(texto)


def test_clean_conserva_fechas_y_etiquetas_normales():
    assert decision._clean("jueves 8 de octubre, 10:00") == "jueves 8 de octubre, 10:00"
    assert decision._clean("2026-10-08") == "2026-10-08"
    assert decision._clean("  varios   espacios\n y saltos ") == "varios espacios y saltos"


# ----------------------------------------------------------------- rastro ---


def test_trace_tope_de_20_pasos_en_orden_de_llamada():
    trace = DecisionTrace()
    for i in range(MAX_STEPS + 5):
        trace.add_step("handoff", {"reason": f"motivo {i}"}, {"ok": True})
    steps = trace.to_dict()["steps"]
    assert len(steps) == MAX_STEPS
    assert steps[0]["summary"] == "pasó a una persona: motivo 0"
    assert steps[-1]["summary"] == f"pasó a una persona: motivo {MAX_STEPS - 1}"


def test_trace_tokens_se_omiten_si_nadie_reporto_uso():
    trace = DecisionTrace()
    trace.add_usage(None)
    assert "tokens" not in trace.to_dict()


def test_trace_suma_tokens_de_todas_las_rondas():
    trace = DecisionTrace()
    trace.add_usage(LlmUsage(input=100, output=10))
    trace.add_usage(None)  # una ronda sin uso no borra lo ya contado
    trace.add_usage(LlmUsage(input=140, output=25))
    assert trace.to_dict()["tokens"] == {"input": 240, "output": 35}


def test_trace_forma_completa_del_contrato():
    trace = DecisionTrace(prompt_version="abc123def456")
    trace.use_model(type("L", (), {"model": "gpt-4o-mini"})())
    trace.add_step("route_out", {}, {"ok": True})
    out = trace.to_dict()
    assert set(out) == {"model", "promptVersion", "steps", "latencyMs"}
    assert out["model"] == "gpt-4o-mini"
    assert out["promptVersion"] == "abc123def456"
    assert out["steps"] == [
        {"tool": "route_out", "summary": "marcó al lead como no calificado", "ok": True}
    ]
    assert isinstance(out["latencyMs"], int) and out["latencyMs"] >= 0


def test_model_id_de_un_cliente_sin_modelo_es_unknown():
    assert decision.model_id(object()) == "unknown"
    assert decision.model_id(type("L", (), {"model": ""})()) == "unknown"


def test_trace_nombre_de_herramienta_inventado_se_sanea():
    trace = DecisionTrace()
    trace.add_step("../../etc/passwd; DROP", {}, {"ok": False, "error": "herramienta desconocida: x"})
    step = trace.to_dict()["steps"][0]
    assert step["tool"] == "....etcpasswdDROP"  # solo [A-Za-z0-9_.-]
    assert step["ok"] is False


def test_trace_si_el_resumen_revienta_cae_a_uno_generico(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("bug en el resumen")

    monkeypatch.setattr(decision, "summarize_step", boom)
    trace = DecisionTrace()
    trace.add_step("propose_slots", {}, {"ok": True})
    assert trace.to_dict()["steps"] == [
        {"tool": "propose_slots", "summary": "usó propose_slots", "ok": True}
    ]
