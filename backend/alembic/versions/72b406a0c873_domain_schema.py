"""domain schema

Revision ID: 72b406a0c873
Revises: 
Create Date: 2026-11-24 15:01:03.802953

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '72b406a0c873'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('customers',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('customer_id_ext', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("customer_id_ext <> '' AND customer_id_ext = btrim(customer_id_ext)", name=op.f('ck_customers_customer_id_ext_normalized')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_customers')),
    sa.UniqueConstraint('customer_id_ext', name='uq_customers_customer_id_ext')
    )
    op.create_table('churn_predictions',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('customer_id', sa.Uuid(), nullable=True),
    sa.Column('batch_id', sa.Uuid(), nullable=True),
    sa.Column('model_version', sa.Text(), nullable=False),
    sa.Column('churn_probability', sa.Float(), nullable=False),
    sa.Column('decision_threshold', sa.Float(), nullable=True),
    sa.Column('risk_band', sa.Text(), nullable=True),
    sa.Column('factors', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('feature_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('feature_schema_version', sa.Text(), nullable=False),
    sa.Column('feature_schema_digest', sa.Text(), nullable=False),
    sa.Column('factors_schema_digest', sa.Text(), nullable=False),
    sa.Column('data_origin', sa.Enum('demo_backfill', 'historical_test', 'prospective', name='data_origin', native_enum=False, create_constraint=True, length=20), nullable=False),
    sa.Column('score_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('feature_as_of', sa.DateTime(timezone=True), nullable=True),
    sa.Column('outcome_window_start', sa.DateTime(timezone=True), nullable=True),
    sa.Column('outcome_window_end', sa.DateTime(timezone=True), nullable=True),
    sa.Column('actual_churn', sa.Boolean(), nullable=True),
    sa.Column('outcome_origin', sa.Enum('demo_backfill', 'historical_test', 'prospective', name='outcome_origin', native_enum=False, create_constraint=True, length=20), nullable=True),
    sa.Column('outcome_recorded_by', sa.Text(), nullable=True),
    sa.Column('outcome_observed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('outcome_recorded_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('outcome_updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('censored_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('outcome_version', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('is_current', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("NOT (data_origin = 'prospective' AND actual_churn IS FALSE) OR (outcome_window_end IS NOT NULL AND COALESCE(outcome_updated_at, outcome_recorded_at) >= outcome_window_end)", name=op.f('ck_churn_predictions_prospective_negative_maturity')),
    sa.CheckConstraint("factors_schema_digest ~ '^[0-9a-f]{64}$'", name=op.f('ck_churn_predictions_factors_schema_digest_hex')),
    sa.CheckConstraint("feature_schema_digest ~ '^[0-9a-f]{64}$'", name=op.f('ck_churn_predictions_feature_schema_digest_hex')),
    sa.CheckConstraint("feature_schema_version <> '' AND feature_schema_version = btrim(feature_schema_version)", name=op.f('ck_churn_predictions_feature_schema_version_normalized')),
    sa.CheckConstraint("jsonb_typeof(factors) = 'object'", name=op.f('ck_churn_predictions_factors_object')),
    sa.CheckConstraint("jsonb_typeof(feature_snapshot) = 'object'", name=op.f('ck_churn_predictions_feature_snapshot_object')),
    sa.CheckConstraint("model_version <> '' AND model_version = btrim(model_version)", name=op.f('ck_churn_predictions_model_version_normalized')),
    sa.CheckConstraint("risk_band IN ('high', 'low')", name=op.f('ck_churn_predictions_risk_band_values')),
    sa.CheckConstraint('(outcome_window_start IS NULL) = (outcome_window_end IS NULL)', name=op.f('ck_churn_predictions_outcome_window_pair')),
    sa.CheckConstraint('(risk_band IS NULL) = (decision_threshold IS NULL)', name=op.f('ck_churn_predictions_risk_band_threshold_coupled')),
    sa.CheckConstraint('actual_churn IS NOT NULL OR (outcome_origin IS NULL AND outcome_recorded_by IS NULL AND outcome_observed_at IS NULL AND outcome_recorded_at IS NULL AND outcome_updated_at IS NULL)', name=op.f('ck_churn_predictions_outcome_metadata_orphan')),
    sa.CheckConstraint('actual_churn IS NULL OR (outcome_origin IS NOT NULL AND outcome_recorded_by IS NOT NULL AND outcome_recorded_at IS NOT NULL)', name=op.f('ck_churn_predictions_outcome_bundle_complete')),
    sa.CheckConstraint('censored_at IS NULL OR actual_churn IS NULL', name=op.f('ck_churn_predictions_censored_without_outcome')),
    sa.CheckConstraint('churn_probability BETWEEN 0 AND 1', name=op.f('ck_churn_predictions_churn_probability_range')),
    sa.CheckConstraint('customer_id IS NOT NULL OR NOT is_current', name=op.f('ck_churn_predictions_sandbox_not_current')),
    sa.CheckConstraint('decision_threshold IS NULL OR decision_threshold BETWEEN 0 AND 1', name=op.f('ck_churn_predictions_decision_threshold_range')),
    sa.CheckConstraint('feature_as_of IS NULL OR feature_as_of <= score_at', name=op.f('ck_churn_predictions_feature_as_of_not_after_score')),
    sa.CheckConstraint('outcome_origin IS NULL OR outcome_origin = data_origin', name=op.f('ck_churn_predictions_outcome_origin_matches_data_origin')),
    sa.CheckConstraint('outcome_updated_at IS NULL OR outcome_updated_at >= outcome_recorded_at', name=op.f('ck_churn_predictions_outcome_update_order')),
    sa.CheckConstraint('outcome_window_start IS NULL OR outcome_window_end > outcome_window_start', name=op.f('ck_churn_predictions_outcome_window_order')),
    sa.CheckConstraint('pg_column_size(factors) <= 16384', name=op.f('ck_churn_predictions_factors_size')),
    sa.CheckConstraint('pg_column_size(feature_snapshot) <= 16384', name=op.f('ck_churn_predictions_feature_snapshot_size')),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], name=op.f('fk_churn_predictions_customer_id_customers'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_churn_predictions'))
    )
    op.create_index('ix_churn_predictions_batch', 'churn_predictions', ['batch_id'], unique=False, postgresql_where=sa.text('batch_id IS NOT NULL'))
    op.create_index('ix_churn_predictions_customer_history', 'churn_predictions', ['customer_id', 'data_origin', 'score_at'], unique=False)
    op.create_index('ix_churn_predictions_origin_score', 'churn_predictions', ['data_origin', 'score_at'], unique=False)
    op.create_index('ix_churn_predictions_worklist', 'churn_predictions', ['data_origin', 'churn_probability', 'id'], unique=False, postgresql_where=sa.text('is_current'))
    op.create_index('uq_churn_predictions_current', 'churn_predictions', ['customer_id', 'data_origin'], unique=True, postgresql_where=sa.text('is_current'))
    op.create_table('customer_services',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('customer_id', sa.Uuid(), nullable=False),
    sa.Column('data_origin', sa.Enum('demo_backfill', 'historical_test', 'prospective', name='data_origin', native_enum=False, create_constraint=True, length=20), nullable=False),
    sa.Column('service_type', sa.Text(), nullable=False),
    sa.Column('attributes', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('valid_from', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('valid_to', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("jsonb_typeof(attributes) = 'object'", name=op.f('ck_customer_services_attributes_object')),
    sa.CheckConstraint("service_type <> '' AND service_type = btrim(service_type)", name=op.f('ck_customer_services_service_type_normalized')),
    sa.CheckConstraint('valid_to IS NULL OR valid_to > valid_from', name=op.f('ck_customer_services_validity_window_order')),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.id'], name=op.f('fk_customer_services_customer_id_customers'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_customer_services'))
    )
    op.create_index('ix_customer_services_customer_history', 'customer_services', ['customer_id', 'data_origin', 'valid_from'], unique=False)
    op.create_index('uq_customer_services_current', 'customer_services', ['customer_id', 'data_origin', 'service_type'], unique=True, postgresql_where=sa.text('valid_to IS NULL'))
    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_index('uq_customer_services_current', table_name='customer_services', postgresql_where=sa.text('valid_to IS NULL'))
    op.drop_index('ix_customer_services_customer_history', table_name='customer_services')
    op.drop_table('customer_services')
    op.drop_index('uq_churn_predictions_current', table_name='churn_predictions', postgresql_where=sa.text('is_current'))
    op.drop_index('ix_churn_predictions_worklist', table_name='churn_predictions', postgresql_where=sa.text('is_current'))
    op.drop_index('ix_churn_predictions_origin_score', table_name='churn_predictions')
    op.drop_index('ix_churn_predictions_customer_history', table_name='churn_predictions')
    op.drop_index('ix_churn_predictions_batch', table_name='churn_predictions', postgresql_where=sa.text('batch_id IS NOT NULL'))
    op.drop_table('churn_predictions')
    op.drop_table('customers')
    # ### end Alembic commands ###
