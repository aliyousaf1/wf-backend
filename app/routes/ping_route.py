from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.core.database import db

pingRouter = APIRouter(
    tags=["Health Check"]
)

@pingRouter.get("/ping")
def ping():
    return {"status": "true", "message": "ALIVE"
}

@pingRouter.get("/databases")
async def get_databases():
    databaseNames = await db.list_database_names()
    return {"status": "true", "databases": databaseNames}

@pingRouter.get("/elements", response_class=HTMLResponse)
def get_elements():
    html_content = """
    <!doctype html>
    <html>
      <head>
        <title>Stoplight Elements</title>
        <script src="https://unpkg.com/@stoplight/elements/web-components.min.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/@stoplight/elements/styles.min.css">
      </head>
      <body>
        <elements-api
          apiDescriptionUrl="/openapi.json"
          router="hash"
          layout="sidebar">
        </elements-api>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@pingRouter.get("/rapidoc", response_class=HTMLResponse)
def get_rapidoc():
    html_content = """
    <!doctype html>
    <html>
      <head>
        <title>RapiDoc</title>
        <script type="module" src="https://unpkg.com/rapidoc/dist/rapidoc-min.js"></script>
      </head>
      <body>
        <rapi-doc 
          spec-url="/openapi.json" 
          theme="light"
          render-style="read"
          primary-color="#ffffff"
          allow-server-selection="false"
          show-header="false"
          hide-schema-titles="false"
          hide-export="true"
          show-header="true"
            nav-bg-color="#eee"
            nav-text-color="#444"
            nav-hover-bg-color="#fff"
          allow-try="true">
        </rapi-doc>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

