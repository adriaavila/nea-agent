"""La capa de perfil: CRM → brief local → perfil mínimo, sin tumbar turnos."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.crm import CrmError
from app.profile import (
    BusinessProfile,
    ProfileProvider,
    profile_from_payload,
    resolve_profile,
)
from app.prompt import build_system_prompt, prompt_version, stable_prompt
from app.state import Conversation


class FakeCrm:
    def __init__(self, payloads):
        self.payloads = list(payloads)  # cada get_profile consume uno
        self.calls = 0

    async def get_profile(self):
        self.calls += 1
        item = self.payloads.pop(0) if self.payloads else None
        if isinstance(item, Exception):
            raise item
        return item


PAYLOAD = {
    "profile": {
        "name": "Sofi",
        "tone": "cálido y directo",
        "instructions": "Vendemos limpiezas dentales. Califica: adultos en la ciudad.",
        "escalationRules": "Urgencias de dolor → humano de inmediato.",
        "greeting": "¡Hola! Soy Sofi, la asistente de la clínica 🦷",
        "activationEnabled": True,
        "activationMessages": ["quiero agendar"],
    },
    "kb": "P: ¿Cuánto cuesta la limpieza?\nR: $800 MXN.",
    "resources": [{"label": "Guía de higiene", "url": "https://example.com/guia"}],
}


def test_profile_from_payload_mapea_todo():
    prof = profile_from_payload(PAYLOAD, default_name="Nea")
    assert prof.agent_name == "Sofi"
    assert prof.tone == "cálido y directo"
    assert prof.escalation_rules and "Urgencias" in prof.escalation_rules
    assert prof.kb_text and "$800" in prof.kb_text
    assert prof.resources == [{"label": "Guía de higiene", "url": "https://example.com/guia"}]
    assert prof.activation_enabled
    assert prof.activation_messages == ("quiero agendar",)
    assert prof.has_knowledge


def test_profile_from_payload_tolerante_a_vacios():
    prof = profile_from_payload({}, default_name="Nea")
    assert prof.agent_name == "Nea"
    assert not prof.has_knowledge


def test_kb_centinela_del_crm_no_cuenta_como_conocimiento():
    # El CRM renderiza el KB vacío como "(knowledge base vacío)" (contrato
    # 009): eso NO es conocimiento — sin instrucciones, el chasis debe seguir
    # advirtiendo el perfil incompleto.
    prof = profile_from_payload(
        {"profile": {"name": "Asistente"}, "kb": "(knowledge base vacío)", "resources": []},
        default_name="Nea",
    )
    assert prof.kb_text is None
    assert not prof.has_knowledge


async def test_provider_cachea_el_perfil_del_crm():
    crm = FakeCrm([PAYLOAD])
    provider = ProfileProvider(crm, ttl=600)
    p1 = await provider.get()
    p2 = await provider.get()
    assert p1.agent_name == "Sofi"
    assert p2 is p1
    assert crm.calls == 1  # el TTL evita martillar al CRM


async def test_provider_cae_al_brief_local_si_crm_404(tmp_path):
    brief = tmp_path / "brief.md"
    brief.write_text("Somos una barbería. Agenda cortes de 30 min.", encoding="utf-8")
    provider = ProfileProvider(FakeCrm([None]), brief_path=str(brief))
    prof = await provider.get()
    assert prof.agent_name == "Nea"
    assert prof.instructions and "barbería" in prof.instructions
    assert prof.has_knowledge


async def test_provider_minimo_sin_crm_ni_brief():
    provider = ProfileProvider(FakeCrm([None]))
    prof = await provider.get()
    assert prof == BusinessProfile(agent_name="Nea")


async def test_provider_sirve_el_ultimo_conocido_si_crm_cae():
    crm = FakeCrm([PAYLOAD, CrmError("caído")])
    provider = ProfileProvider(crm, ttl=0)  # fuerza re-fetch en cada get
    p1 = await provider.get()
    p2 = await provider.get()
    assert p2.agent_name == p1.agent_name == "Sofi"


async def test_resolve_profile_sin_provider_usa_settings():
    ctx = SimpleNamespace(profile=None, settings=SimpleNamespace(agent_name="Max"))
    prof = await resolve_profile(ctx)
    assert prof.agent_name == "Max"


def _conv() -> Conversation:
    return Conversation(id=1, wa_identity="5215550001111", greeted=True)


def test_prompt_compone_chasis_y_negocio():
    prof = profile_from_payload(PAYLOAD, default_name="Nea")
    system = build_system_prompt(profile=prof, context=None, conv=_conv())
    assert "Eres Sofi" in system
    assert "cálido y directo" in system
    assert "limpiezas dentales" in system
    assert "$800" in system
    assert "https://example.com/guia" in system
    assert "route_out" in system
    assert "hostilidad" in system  # el chasis conserva la regla de 3 strikes
    assert "OJO: el negocio aún no configuró" not in system


def test_prompt_minimo_advierte_falta_de_conocimiento():
    system = build_system_prompt(profile=BusinessProfile(), context=None, conv=_conv())
    assert "Eres Nea" in system
    assert "OJO: el negocio aún no configuró" in system
    assert "(sin entradas todavía)" in system


@pytest.mark.parametrize("kb", [None, "  "])
def test_has_knowledge_falso_con_kb_vacio(kb):
    assert not BusinessProfile(kb_text=kb).has_knowledge


# ------------------------------------------------------- prompt_version ---


def test_prompt_version_son_12_hex_y_determinista():
    prof = profile_from_payload(PAYLOAD, default_name="Nea")
    version = prompt_version(prof)
    assert len(version) == 12
    int(version, 16)  # es hex
    assert prompt_version(prof) == version
    # perfil igual construido aparte → misma versión
    assert prompt_version(profile_from_payload(PAYLOAD, default_name="Nea")) == version


def test_prompt_version_ignora_la_parte_por_turno_del_prompt():
    """El bloque "CONTEXTO ACTUAL" (hora, lead, horarios ofrecidos) cambia en
    cada turno y NO entra en la versión: el system prompt completo de dos
    turnos puede diferir y la versión es la misma."""
    from datetime import datetime, timezone

    prof = profile_from_payload(PAYLOAD, default_name="Nea")
    ctx_a = {"contact": {"name": "Ana", "ficha": {"rubro": "dentista"}}}
    ctx_b = {"contact": {"name": "Beto", "ficha": {}}}
    full_a = build_system_prompt(
        profile=prof,
        context=ctx_a,
        conv=_conv(),
        now=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc),
    )
    full_b = build_system_prompt(
        profile=prof,
        context=ctx_b,
        conv=_conv(),
        now=datetime(2026, 10, 2, 17, 30, tzinfo=timezone.utc),
    )
    assert full_a != full_b
    # el prompt completo es exactamente la parte estable + el bloque del turno
    assert full_a.startswith(stable_prompt(prof) + "\n")
    assert full_b.startswith(stable_prompt(prof) + "\n")
    assert "CONTEXTO ACTUAL" not in stable_prompt(prof)


@pytest.mark.parametrize(
    "cambio",
    [
        {"instructions": "Vendemos blanqueamientos."},
        {"tone": "formal"},
        {"agent_name": "Max"},
        {"kb_text": "P: ¿Precio? R: $900"},
    ],
)
def test_prompt_version_cambia_cuando_cambia_el_perfil_del_negocio(cambio):
    from dataclasses import replace

    prof = profile_from_payload(PAYLOAD, default_name="Nea")
    assert prompt_version(replace(prof, **cambio)) != prompt_version(prof)
