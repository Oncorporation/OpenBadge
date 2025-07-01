import json

# Global constant for credentialSchema
CREDENTIAL_SCHEMA = {
    "id": "https://purl.imsglobal.org/spec/ob/v3p0/schema/json/ob_v3p0_achievementcredential_schema.json",
    "type": "JsonSchemaValidator2019"
}

def build_openbadge_metadata(
    credential_id: str,
    subject_id: str,
    issuer: dict,
    valid_from: str,
    achievement: dict,
    alignments: list[dict] = None,
    name: str = None,
    description: str = None,
    image: dict = None,
) -> str:
    """
    Builds Open Badge 3.0 metadata as a JSON-LD string, incorporating required fields and optional skills alignment.

    Args:
        credential_id (str): Unique URI for the credential (e.g., 'urn:uuid:1234').
        subject_id (str): Identifier for the recipient (e.g., DID, email URI).
        issuer (dict): Issuer profile with 'id', 'type', 'name' (e.g., {'id': 'https://example.edu/issuers/565049', 'type': 'Profile', 'name': 'Example University'}).
        valid_from (str): ISO 8601 date when the credential is valid (e.g., '2025-06-25T09:07:00Z').
        achievement (dict): Achievement details, must include 'id', 'type' (['Achievement']), 'name', 'description', 'criteria'.
        alignments (list[dict], optional): List of skill alignments, each with 'targetName', 'targetUrl', etc.
        name (str, optional): Credential name for display.
        description (str, optional): Brief credential description.
        image (dict, optional): Image metadata (e.g., {'id': 'https://example.edu/images/badge.png', 'type': 'Image'}).

    Returns:
        str: JSON-LD string representing the Open Badge 3.0 metadata.

    Example:
        issuer = {
            "id": "https://example.edu/issuers/565049",
            "type": "Profile",
            "name": "Example University"
        }
        achievement = {
            "id": "https://example.edu/achievements/123",
            "type": ["Achievement"],
            "name": "Robotics Badge",
            "description": "Awarded for building a robot",
            "criteria": {"narrative": "Build and demonstrate a robot"}
        }
        alignments = [
            {
                "targetName": "Robotics Skill",
                "targetUrl": "https://skills.example.org/robotics",
                "targetType": "ceasn:Competency"
            }
        ]
        metadata = build_openbadge_metadata(
            credential_id="urn:uuid:1234",
            subject_id="did:example:recipient",
            issuer=issuer,
            valid_from="2025-06-25T09:07:00Z",
            achievement=achievement,
            alignments=alignments,
            name="Robotics Credential"
        )
    """
    # Set default type for the credential
    credential_type = ["VerifiableCredential", "OpenBadgeCredential"]

    # Construct credentialSubject
    credential_subject = {
        "type": ["AchievementSubject"],
        "id": subject_id,
        "achievement": {
            **achievement,
            "alignment": alignments if alignments else []
        }
    }

    # Construct the credential
    credential = {
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://purl.imsglobal.org/spec/ob/v3p0/context-3.0.6.json"
        ],
        "id": credential_id,
        "type": credential_type,
        "issuer": issuer,
        "validFrom": valid_from,
        "credentialSubject": credential_subject,
        "credentialSchema": [CREDENTIAL_SCHEMA]
    }

    # Add optional fields if provided
    if name:
        credential["name"] = name
    if description:
        credential["description"] = description
    if image:
        credential["image"] = image

    # Serialize to JSON-LD string
    return json.dumps(credential, indent=2)