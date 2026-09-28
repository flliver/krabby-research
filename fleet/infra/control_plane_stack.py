"""ControlPlaneStack -- AWS IoT Core fleet control plane for Krabby.

Creates the shared IoT resources that every enrolled krab uses:
- Thing type ``Krab``
- Least-privilege per-thing IoT policy (shadow, teleop signaling, tunnel
  notify, image-upload role assumption)
- Fleet Indexing on registry + Classic Shadow + connectivity
- Lifecycle/presence event configurations (feeds connectivity indexing)
- S3 bucket + IAM role + RoleAlias for agent-reported camera frames --
  ``reported_image`` is an S3 reference, not inline shadow bytes
- Least-privilege IAM user ``krabby-enroll`` for Orin ``krabby enroll``
  (no access key in CDK -- create the key once out-of-band)
- CFN exports ``IotAtsEndpoint`` / ``IotCredentialProviderEndpoint`` /
  ``KrabReportedImagesBucketName`` for enroll / agent / FleetServiceStack

Instance size and EC2 live in FleetServiceStack (later). Secure Tunneling is an
account service -- no per-resource CDK here.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack, custom_resources as cr
from aws_cdk import aws_iam as iam
from aws_cdk import aws_iot as iot
from aws_cdk import aws_s3 as s3
from constructs import Construct

# krabby enroll imports this constant directly to stay in sync with the
# deployed thing type name.
KRAB_THING_TYPE = "Krab"

# Fixed name so krabby enroll can attach the policy by name without
# querying CDK outputs first.
KRAB_DEVICE_POLICY = "KrabDevicePolicy"

# Fixed name so the device policy below can reference it directly to build
# the rolealias ARN, without querying CDK outputs first.
KRAB_IMAGE_ROLE_ALIAS = "KrabImageRoleAlias"

# Fixed name so operators create access keys against a stable IAM user
# without querying CDK outputs first.
KRAB_ENROLL_USER = "krabby-enroll"

# THING/THING_GROUP/THING_TYPE/membership/hierarchy/CERTIFICATE events feed
# Fleet Indexing's connectivity + registry fields; JOB/POLICY/CA_CERTIFICATE
# are unused here.
_IOT_LIFECYCLE_EVENT_CONFIGURATIONS = {
    "THING": {"Enabled": True},
    "THING_GROUP": {"Enabled": True},
    "THING_TYPE": {"Enabled": True},
    "THING_GROUP_MEMBERSHIP": {"Enabled": True},
    "THING_GROUP_HIERARCHY": {"Enabled": True},
    "CERTIFICATE": {"Enabled": True},
    "CA_CERTIFICATE": {"Enabled": False},
    "JOB": {"Enabled": False},
    "JOB_EXECUTION": {"Enabled": False},
    "POLICY": {"Enabled": False},
}


def _device_iot_policy_document() -> dict[str, Any]:
    """Least-privilege policy scoped to the connecting thing name.

    Policy variables ensure a compromised device cannot read/write another
    device's shadow, teleop signaling, or Secure Tunneling notify topic.
    """
    # AWS IoT policy variable, resolved at connect time to the client's
    # thing name -- makes one shared policy document behave per-thing.
    thing = "${iot:Connection.Thing.ThingName}"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                # Device may only connect using its own thing name as clientId --
                # blocks cert reuse under another device's identity.
                "Sid": "ConnectAsSelf",
                "Effect": "Allow",
                "Action": "iot:Connect",
                "Resource": f"arn:aws:iot:*:*:client/{thing}",
            },
            {
                # Classic Shadow: device reports its own state (health/pose/etc).
                "Sid": "ShadowUpdatePublish",
                "Effect": "Allow",
                "Action": "iot:Publish",
                "Resource": [
                    f"arn:aws:iot:*:*:topic/$aws/things/{thing}/shadow/update",
                ],
            },
            {
                # IoT splits Subscribe (attach subscription) from Receive
                # (actual delivery) -- both needed to confirm shadow writes.
                "Sid": "ShadowUpdateSubscribe",
                "Effect": "Allow",
                "Action": "iot:Subscribe",
                "Resource": [
                    f"arn:aws:iot:*:*:topicfilter/$aws/things/{thing}/shadow/update/accepted",
                    f"arn:aws:iot:*:*:topicfilter/$aws/things/{thing}/shadow/update/rejected",
                ],
            },
            {
                "Sid": "ShadowUpdateReceive",
                "Effect": "Allow",
                "Action": "iot:Receive",
                "Resource": [
                    f"arn:aws:iot:*:*:topic/$aws/things/{thing}/shadow/update/accepted",
                    f"arn:aws:iot:*:*:topic/$aws/things/{thing}/shadow/update/rejected",
                ],
            },
            {
                # Lets the device swap its cert for temporary AWS credentials
                # scoped to the image-upload role, to put reported_image
                # frames directly in S3 instead of inline in the shadow.
                "Sid": "AssumeImageUploadRole",
                "Effect": "Allow",
                "Action": "iot:AssumeRoleWithCertificate",
                "Resource": f"arn:aws:iot:*:*:rolealias/{KRAB_IMAGE_ROLE_ALIAS}",
            },
            {
                # SDP/ICE signaling transport between agent and fleet service
                # bridge, carried on the robot's single outbound MQTT link.
                "Sid": "TeleopSignalingPublish",
                "Effect": "Allow",
                "Action": "iot:Publish",
                "Resource": f"arn:aws:iot:*:*:topic/teleop/{thing}/signaling/*",
            },
            {
                "Sid": "TeleopSignalingSubscribe",
                "Effect": "Allow",
                "Action": "iot:Subscribe",
                "Resource": f"arn:aws:iot:*:*:topicfilter/teleop/{thing}/signaling/*",
            },
            {
                "Sid": "TeleopSignalingReceive",
                "Effect": "Allow",
                "Action": "iot:Receive",
                "Resource": f"arn:aws:iot:*:*:topic/teleop/{thing}/signaling/*",
            },
            {
                # Reserved topic AWS pushes OpenTunnel destination tokens to;
                # subscribe-only, device reacts by spawning localproxy.
                "Sid": "SecureTunnelNotifySubscribe",
                "Effect": "Allow",
                "Action": "iot:Subscribe",
                "Resource": f"arn:aws:iot:*:*:topicfilter/$aws/things/{thing}/tunnels/notify",
            },
            {
                "Sid": "SecureTunnelNotifyReceive",
                "Effect": "Allow",
                "Action": "iot:Receive",
                "Resource": f"arn:aws:iot:*:*:topic/$aws/things/{thing}/tunnels/notify",
            },
        ],
    }


class ControlPlaneStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: Any) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Groups every enrolled robot so SearchIndex queries can filter on
        # "all krabs" as a class.
        thing_type = iot.CfnThingType(
            self,
            "KrabThingType",
            thing_type_name=KRAB_THING_TYPE,
            thing_type_properties=iot.CfnThingType.ThingTypePropertiesProperty(
                thing_type_description="Krabby / Krab fleet robot (Orin)",
            ),
        )

        # One shared policy document attached to every device cert --
        # isolation comes from the ${iot:Connection.Thing.ThingName}
        # substitution, not from having separate per-device policies.
        device_policy = iot.CfnPolicy(
            self,
            "KrabDevicePolicy",
            policy_name=KRAB_DEVICE_POLICY,
            policy_document=_device_iot_policy_document(),
        )

        # Account-level singleton: registry + Classic Shadow + connectivity
        # status. CloudFormation has no native resource for Fleet Indexing,
        # so this calls updateIndexingConfiguration via a custom resource.
        indexing_sdk = cr.AwsSdkCall(
            service="Iot",
            action="updateIndexingConfiguration",
            parameters={
                "thingIndexingConfiguration": {
                    "thingIndexingMode": "REGISTRY_AND_SHADOW",
                    "thingConnectivityIndexingMode": "STATUS",
                },
            },
            physical_resource_id=cr.PhysicalResourceId.of("KrabFleetIndexing"),
        )
        indexing = cr.AwsCustomResource(
            self,
            "FleetIndexing",
            on_create=indexing_sdk,
            on_update=indexing_sdk,
            # Account-wide setting with no resource-level ARN to scope to --
            # ANY_RESOURCE is the only option.
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE,
            ),
        )
        # Ordering only, for readable deploy output -- not a functional
        # dependency (both resources are account-level, not parent/child).
        indexing.node.add_dependency(thing_type)

        # Feeds Fleet Indexing's connectivity.connected/timestamp fields --
        # without this, SearchIndex shows stale/empty connectivity data. No
        # Lambda subscriber needed; Fleet Indexing consumes these internally.
        lifecycle_sdk = cr.AwsSdkCall(
            service="Iot",
            action="updateEventConfigurations",
            parameters={"eventConfigurations": _IOT_LIFECYCLE_EVENT_CONFIGURATIONS},
            physical_resource_id=cr.PhysicalResourceId.of("KrabIotLifecycleEvents"),
        )
        lifecycle_events = cr.AwsCustomResource(
            self,
            "EnableIotLifecycleEvents",
            on_create=lifecycle_sdk,
            on_update=lifecycle_sdk,
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE,
            ),
        )
        # Ordering only, for readable deploy output -- both resources are
        # account-level singletons with no real dependency between them.
        lifecycle_events.node.add_dependency(indexing)

        # Per-account, per-region MQTT endpoint. ATS (Amazon Trust Services)
        # is the current cert-chain type required for mTLS connects, vs. the
        # legacy VeriSign type -- krabby enroll/agent need this hostname to
        # connect at all.
        ats_sdk = cr.AwsSdkCall(
            service="Iot",
            action="describeEndpoint",
            parameters={"endpointType": "iot:Data-ATS"},
            physical_resource_id=cr.PhysicalResourceId.of("KrabIotAtsEndpoint"),
        )
        ats_endpoint = cr.AwsCustomResource(
            self,
            "DescribeIotAtsEndpoint",
            on_create=ats_sdk,
            on_update=ats_sdk,
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE,
            ),
        )

        # Distinct from the ATS data endpoint above -- this is the
        # per-account HTTPS endpoint the agent calls (with its device cert)
        # to exchange the image role alias for temporary AWS credentials.
        credential_provider_sdk = cr.AwsSdkCall(
            service="Iot",
            action="describeEndpoint",
            parameters={"endpointType": "iot:CredentialProvider"},
            physical_resource_id=cr.PhysicalResourceId.of("KrabIotCredentialProviderEndpoint"),
        )
        credential_provider_endpoint = cr.AwsCustomResource(
            self,
            "DescribeIotCredentialProviderEndpoint",
            on_create=credential_provider_sdk,
            on_update=credential_provider_sdk,
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE,
            ),
        )

        # Destination for agent-reported camera frames. RETAIN, not the CDK
        # default of DESTROY -- a stack teardown shouldn't silently delete
        # every robot's uploaded images.
        reported_images_bucket = s3.Bucket(
            self,
            "KrabReportedImages",
            lifecycle_rules=[s3.LifecycleRule(expiration=Duration.days(30))],
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Trusted by AWS IoT's Credentials Provider service, not by the
        # device directly -- the device authenticates with its X.509 cert
        # (via AssumeImageUploadRole above), and IoT exchanges that for
        # temporary credentials scoped to this role.
        image_upload_role = iam.Role(
            self,
            "KrabImageUploadRole",
            assumed_by=iam.ServicePrincipal("credentials.iot.amazonaws.com"),
        )
        reported_images_bucket.grant_put(image_upload_role)

        # Fixed name (KRAB_IMAGE_ROLE_ALIAS) so the device policy's
        # AssumeImageUploadRole statement and krabby enroll/agent can
        # reference it without querying CDK outputs first.
        iot.CfnRoleAlias(
            self,
            "KrabImageRoleAlias",
            role_alias=KRAB_IMAGE_ROLE_ALIAS,
            role_arn=image_upload_role.role_arn,
        )

        # Least-privilege IAM user for `krabby enroll` on the Orin -- operator
        # AWS access key is enroll-time only (shell export, never persisted).
        # Access keys are NOT created here: keys must not live in CloudFormation.
        enroll_user = iam.User(
            self,
            "KrabEnrollUser",
            user_name=KRAB_ENROLL_USER,
        )
        enroll_user.add_to_policy(
            iam.PolicyStatement(
                sid="EnrollGetDevicePolicy",
                actions=["iot:GetPolicy"],
                resources=[
                    f"arn:aws:iot:{self.region}:{self.account}:policy/{KRAB_DEVICE_POLICY}",
                ],
            )
        )
        enroll_user.add_to_policy(
            iam.PolicyStatement(
                sid="EnrollThing",
                actions=["iot:DescribeThing", "iot:CreateThing"],
                # Thing names are chosen at enroll time (MAC / --thing-name).
                # Include thingtype/Krab so CreateThing with thingTypeName=Krab
                # stays scoped to this fleet's type in this account/region.
                resources=[
                    f"arn:aws:iot:{self.region}:{self.account}:thing/*",
                    f"arn:aws:iot:{self.region}:{self.account}:thingtype/{KRAB_THING_TYPE}",
                ],
            )
        )
        enroll_user.add_to_policy(
            iam.PolicyStatement(
                sid="EnrollDescribeEndpoint",
                actions=["iot:DescribeEndpoint"],
                # Account endpoint lookup -- no resource-level ARN exists.
                resources=["*"],
            )
        )
        enroll_user.add_to_policy(
            iam.PolicyStatement(
                sid="EnrollCreateCertificate",
                actions=["iot:CreateCertificateFromCsr"],
                # Creates a new cert -- no ARN exists yet at call time.
                resources=["*"],
            )
        )
        enroll_user.add_to_policy(
            iam.PolicyStatement(
                sid="EnrollAttachPolicy",
                actions=["iot:AttachPolicy"],
                # Target is a device cert; include KrabDevicePolicy so attach
                # is limited to that policy name in this account/region.
                resources=[
                    f"arn:aws:iot:{self.region}:{self.account}:cert/*",
                    f"arn:aws:iot:{self.region}:{self.account}:policy/{KRAB_DEVICE_POLICY}",
                ],
            )
        )
        enroll_user.add_to_policy(
            iam.PolicyStatement(
                sid="EnrollAttachThingPrincipal",
                actions=["iot:AttachThingPrincipal"],
                # Resource type for this action is cert; restrict the thing
                # side with iot:ThingArn to this account's things.
                resources=[
                    f"arn:aws:iot:{self.region}:{self.account}:cert/*",
                ],
                conditions={
                    "ArnLike": {
                        "iot:ThingArn": (
                            f"arn:aws:iot:{self.region}:{self.account}:thing/*"
                        ),
                    },
                },
            )
        )

        # Exported so krabby enroll and FleetServiceStack can Fn::ImportValue
        # it instead of re-querying describeEndpoint themselves.
        CfnOutput(
            self,
            "IotAtsEndpoint",
            value=ats_endpoint.get_response_field("endpointAddress"),
            export_name="IotAtsEndpoint",
            description="AWS IoT Core ATS data endpoint for krabby enroll / agent",
        )
        CfnOutput(
            self,
            "IotCredentialProviderEndpoint",
            value=credential_provider_endpoint.get_response_field("endpointAddress"),
            export_name="IotCredentialProviderEndpoint",
            description="AWS IoT Credentials Provider endpoint for agent image uploads",
        )
        CfnOutput(
            self,
            "KrabReportedImagesBucketName",
            value=reported_images_bucket.bucket_name,
            export_name="KrabReportedImagesBucketName",
            description="S3 bucket for agent-reported camera frames; FleetServiceStack needs s3:GetObject here",
        )
        # Not exported -- enroll/agent import these fixed constants directly
        # in code. Output exists for console visibility only.
        CfnOutput(
            self,
            "KrabThingTypeName",
            value=KRAB_THING_TYPE,
            description="IoT thing type for fleet robots",
        )
        CfnOutput(
            self,
            "KrabDevicePolicyName",
            value=KRAB_DEVICE_POLICY,
            description="IoT policy name to attach at enroll time",
        )
        CfnOutput(
            self,
            "KrabImageRoleAliasName",
            value=KRAB_IMAGE_ROLE_ALIAS,
            description="IoT role alias name for agent image uploads",
        )
        CfnOutput(
            self,
            "KrabEnrollUserName",
            value=enroll_user.user_name,
            description="IAM user for krabby enroll; create access key once out-of-band",
        )
        CfnOutput(
            self,
            "KrabEnrollUserArn",
            value=enroll_user.user_arn,
            description="IAM user ARN for krabby enroll",
        )
