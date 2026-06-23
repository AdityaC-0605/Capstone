"""initial predictions table

Revision ID: 0001
Revises:
Create Date: 2026-06-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("prediction_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("api_key_prefix", sa.String(length=20), nullable=False),
        sa.Column("application", sa.JSON(), nullable=False),
        sa.Column("explanation", sa.JSON(), nullable=True),
        sa.Column("sustainability", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_predictions_prediction_id",
        "predictions",
        ["prediction_id"],
        unique=True,
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_index("ix_predictions_prediction_id", table_name="predictions")
    op.drop_table("predictions")
