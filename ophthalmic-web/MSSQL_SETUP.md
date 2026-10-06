# MSSQL Setup

Use this checklist to finish the SQL Server side for Ophthalmic Imaging.

## 1. Install prerequisites

- Install Microsoft SQL Server.
- Install SQL Server Management Studio (SSMS).
- Install ODBC Driver 17 for SQL Server.

## 2. Create the database

In SSMS, create the database named by the shared `MSSQL_DATABASE` environment value.

Example query:

```sql
CREATE DATABASE [<MSSQL_DATABASE value>];
GO
```

## 3. Update environment variables

Edit [`.env`](./.env) and make sure these values are set:

```env
MSSQL_SERVER=localhost\MSSQLSERVER01
MSSQL_DATABASE=<the same permanent database configured for both the app and Alembic>
MSSQL_DRIVER=ODBC Driver 17 for SQL Server
MSSQL_USERNAME=
MSSQL_PASSWORD=
MSSQL_ENCRYPT=no
```

If you use SQL authentication instead of Windows authentication, provide `MSSQL_USERNAME` and `MSSQL_PASSWORD`.

## 4. Install Python dependencies

From `ophthalmic-web/backend`:

```powershell
pip install -r requirements.txt
```

## 5. Start the backend

```powershell
cd .\ophthalmic-web\backend
python -m uvicorn main:app --reload --port 8000
```

On startup, the backend will create:

- `patients`
- `visits`
- `classifications`

## 6. Verify

- `GET http://localhost:8000/health`
- Log in through the frontend
- Run one classification with `MRI Number` set
- Check `GET /patients` and `GET /history/{mri_number}`
