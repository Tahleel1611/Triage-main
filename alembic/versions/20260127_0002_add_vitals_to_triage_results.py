"""Add vitals column to triage_results

Revision ID: 20260127_0002
Revises: 20260127_0001
Create Date: 2026-01-27 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260127_0002'
down_revision: Union[str, None] = '20260127_0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add vitals JSON column to triage_results table."""
    op.add_column(
        'triage_results',
        sa.Column('vitals', sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    """Remove vitals column from triage_results table."""
    op.drop_column('triage_results', 'vitals')
