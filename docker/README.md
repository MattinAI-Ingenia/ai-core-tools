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
sin IdP externo.

(El modo `FAKE` está retirado; el endpoint de dev-login ya no existe.)

Quién es admin (`OMNIADMIN`) no se guarda en la fila del usuario: se calcula en
cada request comparando su email contra la lista `AICT_OMNIADMINS` del `.env`.

**En el primer arranque el backend provisiona automáticamente** las cuentas de
`AICT_OMNIADMINS` que aún no existan y publica en el log (nivel WARNING) un
enlace de alta de contraseña **de un solo uso** y con caducidad
(`services/auth/omniadmin_bootstrap.py`):

```bash
docker compose logs backend | grep -A5 "omniadmin_bootstrap: NEW"
```

Abre ese enlace (`.../set-password?token=...`) para fijar la contraseña
inicial. Una vez usado, o si caducó, ya no vuelve a aparecer — usa entonces la
ruta de recuperación de abajo.

Si prefieres no depender del log (o necesitas recrear la contraseña porque el
enlace ya se usó/caducó), el script de seeding la fija directamente:

```bash
# 1. En .env: AICT_OMNIADMINS=tu@email.com
# 2. Crear/recuperar el usuario con contraseña (dentro del contenedor backend):
docker compose exec backend python -m utils.seed_dev_users --yes \
  --users "tu@email.com:Tu Nombre:TuPasswordSegura123!"
# o con el wrapper:
./seed-users.sh --users "tu@email.com:Tu Nombre:TuPasswordSegura123!"
```

Luego logueas en `http://localhost/` (o la URL del servidor) con ese
email/password — al coincidir con `AICT_OMNIADMINS` obtiene privilegios de
omniadmin automáticamente, sin pasos adicionales.

El script es idempotente (usuarios existentes no se tocan si no les pasas
contraseña) y corre **dentro del contenedor backend**, reutilizando su config
de BD — no hace falta Python ni acceso directo a Postgres en el host.

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
docker compose exec -T postgres psql -U mattin -d mattin_ai <<EOF
INSERT INTO "User" (email, name, create_date, is_active, auth_method, email_verified) VALUES
  ('user1@cliente.com', 'User 1', NOW(), true, 'dev', true),
  ('user2@cliente.com', 'User 2', NOW(), true, 'dev', true)
ON CONFLICT DO NOTHING;
EOF
```

## Varios Mattin a la vez (multi-entorno local)

Trabajar contra varios proyectos que corren **versiones distintas** de Mattin
—la demo que enseñas a clientes, la funcionalidad que estás implementando, la
PR de otra persona, la versión pineada de un cliente— exige un stack por
versión. No es preferencia: `backend/Dockerfile` arranca con
`alembic upgrade head`, así que **dos versiones sobre el mismo volumen de
Postgres se migran la BD la una a la otra**, y el downgrade no siempre
recupera los datos.

La unidad de aislamiento es el **project name** de Compose, que prefija
contenedores, redes y volúmenes
([docs](https://docs.docker.com/compose/how-tos/project-name/)). Cada entorno
es un fichero en `envs/` que se carga encima de `envs/base.env`:

```
docker/envs/
  base.env          # secretos y config común a todos (gitignored)
  demo.env          # develop del registry  → mattin-demo    :8080
  dev.env           # tu working tree       → mattin-dev     :8081
  review.env        # la rama de otro       → mattin-review  :8082
  dibal.env         # cliente pineado       → mattin-dibal   :8090
  _templates/       # plantillas de `mattin.ps1 new`
```

Los `.env` están gitignored; se versionan sus `.env.example`. Para empezar en
una máquina nueva: copiar cada `.example` a `.env` y rellenar `base.env`.

### El comando

```powershell
cd docker
.\mattin.ps1 ls                       # qué entornos hay y cuáles están vivos
```

| Comando | Para qué |
|---|---|
| `up <e>` | Arranca con la imagen que ya haya |
| `rebuild <e>` | Build desde el working tree actual → entornos de trabajo |
| `update <e>` | `pull` + recrear → demo y clientes |
| `stop` / `start` | Libera RAM / vuelve, **conservando datos** |
| `down <e>` | Borra contenedores, conserva volúmenes |
| `destroy <e>` | Borra también los volúmenes (pide confirmación) |
| `new <n> [-Type feature\|cliente]` | Crea un entorno con puerto libre asignado |
| `rm <n>` | `destroy` + borra su `envs/<n>.env` |
| `logs` / `ps` / `exec` / `psql` / `open` | Operación del día a día |
| `invite <e>` / `setpass <e> <email> <pw>` | Entrar la primera vez (ver abajo) |

Todo esto es Compose plano por debajo; el script solo evita repetir los
`--env-file` y añade unas cuantas comprobaciones. El equivalente crudo:

```bash
docker compose --env-file envs/base.env --env-file envs/dev.env up -d
```

El segundo `--env-file` sobreescribe al primero
([precedencia](https://docs.docker.com/compose/how-tos/environment-variables/envvars-precedence/)),
y `--env-file` **sustituye** al `.env` por defecto del directorio: los entornos
locales y el despliegue single-host de `docker/.env` no se mezclan.

### Los tres casos de uso

**Demo a cliente** — `demo` corre la imagen `:develop` que publica CI, nunca un
build tuyo. Sus datos (apps, agentes, silos de demostración) persisten entre
actualizaciones.

```powershell
.\mattin.ps1 update demo     # a la última develop
.\mattin.ps1 open demo
```

**Probar una funcionalidad** — `dev` para lo tuyo, `review` para la rama de
otra persona. Están separados a propósito: revisar una PR ajena no debería
obligarte a tirar tu propia BD de trabajo.

```powershell
git switch feature/lo-que-sea
.\mattin.ps1 rebuild dev

git fetch origin pull/231/head:pr-231 && git switch pr-231
.\mattin.ps1 rebuild review
.\mattin.ps1 destroy review          # al terminar
```

Destruir `review` al acabar no es limpieza, es lo que hace honesta la siguiente
revisión: sobre BD vacía compruebas que *sus* migraciones corren de cero, no
que corren sobre los restos de la PR anterior.

Si necesitas más de uno a la vez (dos PRs en paralelo), `new` los crea con
puerto libre:

```powershell
.\mattin.ps1 new pr-231              # asigna 8083, IMAGE_TAG=local-pr-231
.\mattin.ps1 rebuild pr-231
.\mattin.ps1 rm pr-231               # cuando sobre
```

**Reproducir la versión de un cliente** — un entorno por cliente, con
`IMAGE_TAG` pineado al tag exacto que corre en producción.

```powershell
.\mattin.ps1 new acme -Type cliente  # asigna 8091; editar IMAGE_TAG
.\mattin.ps1 up acme
```

> `dibal.env` es la plantilla trabajada de este patrón, pero **no** gobierna el
> despliegue de Dibal que ya existe en esta máquina: aquel es otro Compose
> project (`dibal-satia`, con su Redis, worker y panel propios). `up dibal`
> levanta una reproducción local y vacía de la versión de Mattin que corre allí.

### Entrar la primera vez

Con `AICT_LOGIN=LOCAL`, el backend provisiona en el primer arranque las cuentas
de `AICT_OMNIADMINS` y publica un enlace de alta de contraseña —de un solo uso—
en el log (`services/auth/omniadmin_bootstrap.py`):

```powershell
.\mattin.ps1 invite dev              # saca el enlace del log
```

Si el enlace ya se usó o caducó, la ruta de recuperación que soporta el propio
backend (`backend/utils/seed_dev_users.py`, funciona también sobre imágenes del
registry):

```powershell
.\mattin.ps1 setpass dev tu@correo.com 'UnaPasswordBuena123!'
```

### Trampas conocidas

- **`IMAGE_TAG` propio por entorno no es opcional.** El servicio `backend`
  etiqueta lo que construye como `mattinai-backend:${IMAGE_TAG}`. Si un entorno
  de trabajo usara `IMAGE_TAG=develop`, un `--build` machacaría en local la
  imagen `:develop` de `demo`. `rebuild` se niega a correr sobre un tag
  publicado precisamente por esto.
- **Ficheros de repositorios.** Por defecto el compose los monta desde
  `../backend/data` (bind mount), que sería **compartido por todos los
  entornos**. Los `envs/*.env` ponen `BACKEND_DATA=backend-data` para usar un
  volumen nombrado, que Compose prefija con el project name. Los despliegues de
  cliente ya existentes no cambian: sin esa variable sigue el bind mount.
  Para llevarte lo que ya hubiera en `backend/data`:
  ```bash
  docker run --rm -v "$PWD/../backend/data:/from" -v mattin-dev_backend-data:/to \
    alpine sh -c "cp -a /from/. /to/"
  ```
- **OpenSandbox.** `opensandbox/sandbox.toml` fija el nombre de la red en
  `network_mode`. El compose lo expone como `SANDBOX_NETWORK_NAME`, pero para
  levantar el perfil `opensandbox` en más de un stack a la vez cada uno necesita
  además su propia copia de `sandbox.toml`. Con un solo stack usándolo, nada
  que hacer.
- **Puertos.** `up` aborta si el `HTTP_PORT` del entorno ya está ocupado por
  otro, en vez de dejar el stack a medio levantar. `new` elige puerto libre solo
  (8083+ para trabajo, 8090+ para clientes).
- **RAM.** No hace falta tenerlos todos arriba. Cada stack son 5 contenedores;
  `stop` los apaga conservando los datos y `start` los devuelve en segundos. Con
  `VECTOR_DB_TYPE=PGVECTOR` puedes parar además su Qdrant:
  `.\mattin.ps1 stop dev qdrant`.

## Acceso a la base de datos desde fuera

Postgres está publicado en el host en el puerto `5434` (→ 5432 del contenedor).
Dos formas de acceder:

1. **Desde el servidor, psql del contenedor** (rápido):
   ```bash
   docker compose exec postgres psql -U mattin -d mattin_ai
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
