from urllib.parse import quote
from pathlib import PurePosixPath

# Maps file extension to SharePoint Online viewer prefix
_VIEWER_PREFIX = {
    ".docx": ":w:",
    ".doc":  ":w:",
    ".xlsx": ":x:",
    ".xls":  ":x:",
    ".pptx": ":p:",
    ".ppt":  ":p:",
}

def build_sharepoint_file_url(tenant: str, folder_path: str, file_name: str) -> str:
    """
    Builds a SharePoint URL that opens the file in the browser (Word/Excel/PowerPoint Online)
    instead of downloading it.

    Uses the SharePoint Web Viewer format:
        https://{tenant}.sharepoint.com/{viewer_prefix}/r{encoded_path}?web=1

    The ?web=1 parameter forces the online viewer. The viewer prefix (:w:, :x:, :p:)
    is determined by the file extension. Falls back to a plain URL for unsupported types.

    Example:
        build_sharepoint_file_url(
            tenant="ycopvtltd",
            folder_path="/sites/RAGSys_RADAR/Shared Documents/Batch Issues",
            file_name="Batch Issues April 1.docx"
        )
        # -> https://ycopvtltd.sharepoint.com/:w:/r/sites/RAGSys_RADAR/Shared%20Documents/...?web=1
    """
    suffix = PurePosixPath(file_name).suffix.lower()
    viewer_prefix = _VIEWER_PREFIX.get(suffix)

    full_path = f"{folder_path}/{file_name}"
    encoded = quote(full_path)

    if viewer_prefix:
        return f"https://{tenant}.sharepoint.com/{viewer_prefix}/r{encoded}?web=1"
    else:
        # For unsupported types (e.g. .txt, .md), fall back to the direct path
        return f"https://{tenant}.sharepoint.com{encoded}"


if __name__ == "__main__":
    file_url = build_sharepoint_file_url(
        tenant="siammakrogroup",
        folder_path="/sites/RAGSys_RADAR/Shared Documents/Batch Issues",
        file_name="Batch Issues April 1.docx"
    )
    print(file_url)
