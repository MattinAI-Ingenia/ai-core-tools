# Despliegue Docker single-host

Despliegue con **Caddy** como reverse proxy. Sirve tanto para **desarrollo local** en el portátil como para **servidores cliente** (demos, formaciones, pilotos). La única diferencia entre escenarios es el contenido del `.env`.

## Arquitectura

```
Usuario
  │
  ▼
[ Caddy :80 ]  ← único puerto pensado para acceso externo
  │
  ├── /internal/*, /public/*, /mcp/*, /docs/*, /scalar, /static/*, /openapi-*.json, /health
  │        └─► backend:8000
  │
  └── resto (SPA)
           └─► frontend:80

Red interna de Docker (mattin-network):
  postgres:5432 ← publicado también en localhost:5434 (acceso directo en dev)
  qdrant:6333   ← sin publicar al host
  neo4j:7687    ← sin publicar al host (grafo de conocimiento de LightRAG)
```

LightRAG (RAG con grafo de conocimiento) usa Neo4j como backend de grafo; se levanta
siempre junto al resto del stack. Variables en `.env.example` (sección "LightRAG").

Frontend y backend viajan por el mismo origen → **sin CORS**, sin necesidad de rebuildear el frontend entre entornos (`VITE_API_BASE_URL=""` usa rutas relativas).

## Dos formas de obtener las imágenes

| | Opción A — Pull desde GHCR | Opción B — Build en local |
|---|---|---|
| Cuándo | Cliente, demos, CI de producción | Dev local con cambios de código |
| Qué hace | Descarga las imágenes prebuildeadas del registry público | Construye `backend` y `frontend` desde los Dockerfiles del repo |
| Ventaja | Rápido, determinista, no necesita código fuente | Incluye tus cambios locales sin publicar |
| Comando | `docker compose pull && docker compose up -d` | `docker compose up -d --build` |

Las imágenes publicadas viven en:
- `ghcr.io/lksnext-ai-lab/mattinai-backend:${IMAGE_TAG}`
- `ghcr.io/lksnext-ai-lab/mattinai-frontend:${IMAGE_TAG}`

El tag por defecto es `develop` (último build de la rama `develop`). En servidores de cliente se recomienda **pinear un SHA** (`IMAGE_TAG=sha-c1feaaf`) para evitar actualizaciones accidentales al hacer `docker compose pull`.

## Uso

### A. Despliegue tirando de GHCR (recomendado para cliente)

```bash
cd docker
cp .env.example .env
# Editar .env:
#   AICT_PUBLIC_URL=http://<ip-o-dominio>
#   DATABASE_PASSWORD=<robusta>   # [REQUERIDO] el compose falla si está vacío
#   SECRET_KEY=<hex aleatorio>    # [REQUERIDO] el compose falla si está vacío
#   AICT_OMNIADMINS=<emails del cliente>
#   OPENAI_API_KEY=<...>
#   IMAGE_TAG=sha-<commit>   # o "develop" para el último
docker compose pull backend frontend
docker compose up -d
```

Accede a `http://<ip-del-servidor>/` (o `http://localhost/` en local).
Pide al administrador de red del cliente que abra el **80/tcp** hacia el servidor.

> Si las imágenes del registry son privadas, autentícate antes con
> `docker login ghcr.io -u <usuario-github>` usando un Personal Access Token
> con scope `read:packages`.

### B. Dev local con cambios de código

```bash
cd docker
cp .env.example .env
# Editar .env: DATABASE_PASSWORD y SECRET_KEY son [REQUERIDOS] (el compose falla
# si están vacíos); además OPENAI_API_KEY y AICT_OMNIADMINS
docker compose up -d --build
```

`--build` reconstruye las imágenes desde los Dockerfiles y las etiqueta como
`ghcr.io/lksnext-ai-lab/mattinai-backend:develop` (queda local, no se publica).

Accede a `http://localhost/`.

## Comandos habituales

```bash
# Ver estado
docker compose ps

# Logs en vivo (todos)
docker compose logs -f

# Logs solo del backend
docker compose logs -f backend

# Reiniciar un servicio concreto
docker compose restart backend

# Parar
docker compose down

# Parar y BORRAR volúmenes (¡se pierden datos!)
docker compose down -v

# Rebuild tras cambios de código (opción B)
docker compose up -d --build

# Actualizar a la última imagen publicada (opción A)
docker compose pull backend frontend
docker compose up -d
```

## Primer login y crear un admin (modo LOCAL)

`AICT_LOGIN=LOCAL` (el default) gestiona usuarios con email+password propios,
sin IdP externo. No hay ningún usuario admin de fábrica: hay que crearlo y
**darle contraseña explícitamente**, si no, queda creado pero sin forma de
loguearse.

Quién es admin (`OMNIADMIN`) no se guarda en la fila del usuario: se calcula en
cada request comparando su email contra la lista `AICT_OMNIADMINS` del `.env`.
Por tanto, para tener un admin funcional hacen falta las dos cosas:

1. Su email está en `AICT_OMNIADMINS` (`.env`).
2. Existe como `User` en la base de datos **con contraseña**.

```bash
# 1. En .env: AICT_OMNIADMINS=tu@email.com
# 2. Crear el usuario con contraseña (dentro del contenedor backend):
docker compose exec backend python -m utils.seed_dev_users --yes \
  --users "tu@email.com:Tu Nombre:TuPasswordSegura123!"
# o con el wrapper:
./seed-users.sh --users "tu@email.com:Tu Nombre:TuPasswordSegura123!"
```

Luego logueas en `http://localhost/` (o la URL del servidor) con ese
email/password — al coincidir con `AICT_OMNIADMINS` obtiene privilegios de
omniadmin automáticamente, sin pasos adicionales.

El script es idempotente (usuarios existentes no se tocan) y corre **dentro
del contenedor backend**, reutilizando su config de BD — no hace falta Python
ni acceso directo a Postgres en el host.

Otros usos:

```bash
# Usuarios por defecto (admin@example.com, etc. — se crean SIN contraseña,
# solo sirve si luego usas el flujo de "olvidé mi contraseña")
docker compose exec backend python -m utils.seed_dev_users --yes

# Ver qué usuarios se crearían sin escribir nada
docker compose exec backend python -m utils.seed_dev_users --list
```

> El script se niega a correr si `AICT_LOGIN` no es `LOCAL` (evita crear
> usuarios con password en un despliegue OIDC). `--force` lo salta a propósito.

Para sembrar usuarios de forma declarativa al desplegar, define
`AICT_DEV_SEED_USERS` en el `.env` (ver `.env.example`) y lanza el script sin
`--users`.

### Alternativa: SQL directo

Si prefieres no usar el script, puedes insertar directamente vía `psql`:

```bash
docker exec -i mattin-postgres psql -U mattin -d mattin_ai <<EOF
INSERT INTO "User" (email, name, create_date, is_active, auth_method, email_verified) VALUES
  ('user1@cliente.com', 'User 1', NOW(), true, 'dev', true),
  ('user2@cliente.com', 'User 2', NOW(), true, 'dev', true)
ON CONFLICT DO NOTHING;
EOF
```

## Acceso a la base de datos desde fuera

Postgres está publicado en el host en el puerto `5434` (→ 5432 del contenedor).
Dos formas de acceder:

1. **Desde el servidor, psql del contenedor** (rápido):
   ```bash
   docker exec -it mattin-postgres psql -U mattin -d mattin_ai
   ```

2. **DBeaver/pgAdmin directo**: conecta a `<host-o-ip>:5434` con las
   credenciales `DATABASE_USER`/`DATABASE_PASSWORD` del `.env`. En un servidor
   de cliente, si no quieres exponer el 5434 a internet, usa un túnel SSH
   (`ssh -L 5434:localhost:5434 usuario@<ip-servidor>`) en vez de abrir el
   puerto en el firewall.

## Paso a HTTPS

Cuando el cliente tenga dominio interno y abra el 443:

1. Edita el `Caddyfile`:
   ```
   mattinai.cliente.local {
       tls internal    # cert de la CA interna de Caddy (autofirmado)
       encode zstd gzip
       @backend path /internal/* /public/* /mcp/* /docs/* /scalar /openapi-*.json /static/* /health
       handle @backend { reverse_proxy backend:8000 }
       handle { reverse_proxy frontend:80 }
   }
   ```
2. En el compose, publica también el 443:
   ```yaml
   caddy:
     ports:
       - "80:80"
       - "443:443"
   ```
3. Si el cliente tiene PKI corporativa, monta el cert del cliente en el contenedor y sustituye `tls internal` por `tls /etc/caddy/cert.pem /etc/caddy/key.pem`.

## Utilities aisladas

En [`utilities/`](./utilities/) hay compose files para servicios aislados que no forman parte del stack principal (p. ej. Qdrant standalone con su web UI para experimentación).

## Para producción seria (K8s)

Este despliegue es para **single-host**: dev local, POCs, pilotos, demos de cliente. Para producción con alta disponibilidad, múltiples réplicas, TLS automático con Let's Encrypt, backups gestionados, etc., usar los Helm charts en el repo `mattinai-infra`.
