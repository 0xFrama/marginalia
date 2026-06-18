from patient.schema import Base


def create_all(engine):
    Base.metadata.create_all(engine)
