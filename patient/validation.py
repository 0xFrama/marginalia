import sqlglot
from sqlglot import ParseError, exp


class UnsafeSQLError(Exception):
    pass


ALLOWED_TABLES = {
    "v_patient_summary",
    "v_conditions",
    "v_medications",
    "v_observations",
}


def validate(sql: str, dialect: str = "sqlite") -> str:
    """Return safe SQL or raises UnsafeSQLError"""
    statements = [stmt for stmt in sqlglot.parse(sql, read=dialect) if stmt]
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"Rejected: Expected exactly 1 SQL statement, found {len(statements)}"
        )
    statement = statements[0]
    if statement.key != "select":
        raise UnsafeSQLError(
            f"Rejected: Unauthorized query type '{statement.key.upper()}'. Only SELECT queries are permitted."
        )
    for table in statement.find_all(exp.Table):
        if table.name not in ALLOWED_TABLES:
            raise UnsafeSQLError(
                f"Rejected: Only allowed tables: v_patient_summary, v_observations, v_medications, v_conditions. Table found: {table}."
            )

    if statement.args.get("limit") is None:
        statement = statement.limit(100)

    return statement.sql()


if __name__ == "__main__":
    sql = "SELECT * FROM my_table WHERE patient_id = Gino"
    print(validate(sql))
