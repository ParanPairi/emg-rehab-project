from fastapi import APIRouter

from schemas.emg_schema import PrescriptionRequest
from services.prescription_service import create_rehab_prescription


router = APIRouter()


@router.post("/api/prescription")
def create_prescription(data: PrescriptionRequest):
    prescription = create_rehab_prescription(
        patient_name=data.patient_name,
        fatigue_slope=data.fatigue_slope,
        comp_ratio=data.comp_ratio,
        days_to_target=data.days_to_target,
    )

    return {
        "patient_name": data.patient_name,
        "fatigue_slope": data.fatigue_slope,
        "comp_ratio": data.comp_ratio,
        "days_to_target": data.days_to_target,
        "prescription": prescription,
    }