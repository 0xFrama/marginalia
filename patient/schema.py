from datetime import datetime, date

from sqlalchemy import DateTime, ForeignKey, Integer, Integer, String, Date, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Patients(Base):
    __tablename__ = "patients"

    patient_id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    dob: Mapped[date] = mapped_column(Date)
    mrn: Mapped[str] = mapped_column(String(30))
    sex: Mapped[str] = mapped_column(String(30))


class Conditions(Base):
    __tablename__ = "conditions"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"))
    code: Mapped[str] = mapped_column(String(30))
    description: Mapped[str] = mapped_column((String(200)))
    onset_date: Mapped[date] = mapped_column(Date)


class Medications(Base):
    __tablename__ = "medications"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"))
    drug_name: Mapped[str] = mapped_column(String(30))
    dose: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(30))
    started_on: Mapped[date] = mapped_column((Date))


class Observations(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"))
    code: Mapped[str] = mapped_column(String(30))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(30))
    taken_at: Mapped[datetime] = mapped_column(DateTime)


class Encounters(Base):
    __tablename__ = "encounters"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"))
    reason: Mapped[str] = mapped_column(String(30))
    encounter_date: Mapped[date] = mapped_column(Date)


class Allergies(Base):
    __tablename__ = "allergies"

    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.patient_id"))
    substance: Mapped[str] = mapped_column(String(30))
    reaction: Mapped[str] = mapped_column(String(30))


class Billing(Base):
    __tablename__ = "billing"

    bill_id: Mapped[int] = mapped_column(primary_key=True)
    pat_ref: Mapped[str] = mapped_column(String(30))
    amt_cents: Mapped[int] = mapped_column(Integer)
    dx_cd: Mapped[str] = mapped_column(String(30))
    ts: Mapped[datetime] = mapped_column(DateTime)


class Appointments(Base):
    __tablename__ = "appointments"

    appt_id: Mapped[int] = mapped_column(primary_key=True)
    pat_ref: Mapped[str] = mapped_column(String(30))
    prov_id: Mapped[int] = mapped_column(Integer)
    slot_ts: Mapped[datetime] = mapped_column(DateTime)
    status_cd: Mapped[str] = mapped_column(String(30))


class RawHl7Staging(Base):
    __tablename__ = "raw_hl7_staging"

    msg_id: Mapped[int] = mapped_column(primary_key=True)
    rcvd_ts: Mapped[datetime] = mapped_column(DateTime)
    raw_msg: Mapped[str] = mapped_column(String(200))


class AuditLog(Base):
    __tablename__ = "audit_log"

    evt_id: Mapped[int] = mapped_column(primary_key=True)
    usr: Mapped[str] = mapped_column(String(30))
    act: Mapped[str] = mapped_column(String(30))
    obj_ref: Mapped[str] = mapped_column(String(30))
    ts: Mapped[datetime] = mapped_column(DateTime)
