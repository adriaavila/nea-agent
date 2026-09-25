-- 003_org.sql — organización opcional para el modo de despacho multi-tenant.
--
-- Con un solo negocio (webhook de Meta + relay, de siempre) `organization_id`
-- es NULL: es exactamente el comportamiento de antes de esta migración, así
-- que una instalación de un solo negocio no cambia en nada.
--
-- Con un Vocero multitenant, el CRM le despacha turnos de VARIAS
-- organizaciones a esta misma Nea (`POST /dispatch`). `wa_identity` era
-- UNIQUE a secas: correcto con un solo negocio (una identidad de WhatsApp es
-- una persona, y una persona es una conversación), pero con varias
-- organizaciones deja de serlo — el MISMO lead puede escribirle a dos
-- negocios distintos, y con la unicidad global las dos conversaciones se
-- fusionarían en una (el historial de un negocio se filtraría al prompt del
-- otro).
--
-- La clave pasa a ser (organización, identidad). NULL sigue siendo el
-- namespace legacy: en Postgres, dos NULL nunca son iguales entre sí dentro
-- de un índice único, así que UNIQUE(organization_id, wa_identity) con
-- organization_id NULL permitiría duplicados justo en la instalación de
-- siempre — el índice usa COALESCE(organization_id, '') para tratar TODOS los
-- NULL como un mismo namespace ('').
--
-- Idempotente (re-correr en cada arranque es seguro, Constitución III).
--
-- ## Rollback (manual — nada de esto corre solo)
--
-- Si hay que volver al código de ANTES de esta migración (el que hace
-- `ON CONFLICT (wa_identity)` a secas — ese `ON CONFLICT` truena en cuanto el
-- constraint de abajo ya no existe, así que un rollback de CÓDIGO sin
-- rollback de MIGRACIÓN no arranca):
--
--   1. Verificar que la promesa siga siendo cierta (ninguna identidad
--      compartida entre organizaciones — si esto devuelve filas, el rollback
--      pierde información: dos conversaciones de organizaciones distintas se
--      fusionarían en una al recrear el UNIQUE(wa_identity) global):
--        SELECT wa_identity FROM bot_conversation
--        GROUP BY wa_identity HAVING COUNT(DISTINCT organization_id) > 1;
--   2. DROP INDEX IF EXISTS uq_bot_conversation_org_identity;
--   3. ALTER TABLE bot_conversation
--        ADD CONSTRAINT bot_conversation_wa_identity_key UNIQUE (wa_identity);
--   4. Las columnas organization_id pueden quedarse — el código viejo nunca
--      las lee ni las escribe, así que no hace falta un DROP COLUMN.
--
-- IMPORTANTE: nea-santorini (single-tenant, en producción) corre la rama
-- `main`, NO esta rama (`feat/dispatch-multiorg` / `codex/nea-production`)
-- todavía. Esta migración solo se aplica cuando ESE despliegue se actualice a
-- un commit que la incluya — hasta entonces, este rollback es documentación,
-- no una operación pendiente contra producción.

ALTER TABLE bot_conversation ADD COLUMN IF NOT EXISTS organization_id TEXT;
ALTER TABLE pending_send ADD COLUMN IF NOT EXISTS organization_id TEXT;

-- El nombre lo puso Postgres al declarar UNIQUE en la columna (001_init).
ALTER TABLE bot_conversation DROP CONSTRAINT IF EXISTS bot_conversation_wa_identity_key;

CREATE UNIQUE INDEX IF NOT EXISTS uq_bot_conversation_org_identity
  ON bot_conversation ((COALESCE(organization_id, '')), wa_identity);

-- Útil para el barrido de followups/pending_send por organización (no es de
-- unicidad, solo acelera los workers de fondo cuando hay muchas orgs).
CREATE INDEX IF NOT EXISTS idx_bot_conversation_org
  ON bot_conversation (organization_id);
CREATE INDEX IF NOT EXISTS idx_pending_send_org
  ON pending_send (organization_id);
