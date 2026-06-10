"""Initial migration

Revision ID: de138fcb00a8
Revises: 
Create Date: 2026-06-10 16:06:57.780516

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'de138fcb00a8'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create prediction_logs table."""
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False, index=True),
        sa.Column("input_features", sa.JSON(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("risk_band", sa.String(length=16), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False),
        sa.Column("prediction_time_ms", sa.Float(), nullable=True),
        sa.Column("error", sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop prediction_logs table."""
    op.drop_table("prediction_logs")
