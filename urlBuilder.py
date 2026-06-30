from urllib.parse import quote

def build_sharepoint_file_url(tenant: str, folder_path: str, file_name: str) -> str:
    """
    Builds a SharePoint direct file URL from tenant, folder path, and file name.

    Example:
        build_sharepoint_file_url(
            tenant="ycopvtltd",
            folder_path="/sites/RAGSys_RADAR/Shared Documents/Batch Issues",
            file_name="Batch Issues April 1.docx"
        )
        # -> https://ycopvtltd.sharepoint.com/sites/RAGSys_RADAR/Shared%20Documents/...
    """
    full_path = f"{folder_path}/{file_name}"
    encoded = quote(full_path)
    return f"https://{tenant}.sharepoint.com{encoded}"


if __name__ == "__main__":
    file_url = build_sharepoint_file_url(
        tenant="ycopvtltd",
        folder_path="/sites/RAGSys_RADAR/Shared Documents/Batch Issues",
        file_name="CRM_ADW_WC_CUSTOMER_DH.docx"
    )
    print(file_url)