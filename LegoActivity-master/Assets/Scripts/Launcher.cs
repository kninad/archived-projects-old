using UnityEngine;
using UnityEngine.UI;
using Photon.Pun;
using Photon.Realtime;
using System.Collections.Generic;
using UnityEngine.SceneManagement;

public class Launcher : MonoBehaviourPunCallbacks
{
    #region Private Serializable Fields

    [SerializeField]
    private Canvas mainCanvas;
    [SerializeField]
    private Canvas newRoomCanvas;
    [SerializeField]
    private Canvas connectingCanvas;
    [SerializeField]
    private Canvas playerNameCanvas;
    [SerializeField]
    private InputField buildPrompt;
    [SerializeField]
    private InputField playerName;
    [SerializeField]
    private GameObject joinButtonPanel;
    [SerializeField]
    private GameObject joinButtonPrefab;
    [SerializeField]
    private Text playerNameDisplay;


    #endregion


    #region Private Fields


    /// <summary>
    /// This client's version number. Users are separated from each other by gameVersion (which allows you to make breaking changes).
    /// </summary>
    string gameVersion = "1";

    


    #endregion


    #region MonoBehaviour CallBacks


    /// <summary>
    /// MonoBehaviour method called on GameObject by Unity during early initialization phase.
    /// </summary>
    void Awake()
    {
        // #Critical
        // this makes sure we can use PhotonNetwork.LoadLevel() on the master client and all clients in the same room sync their level automatically
        PhotonNetwork.AutomaticallySyncScene = true;
    }


    /// <summary>
    /// MonoBehaviour method called on GameObject by Unity during initialization phase.
    /// </summary>
    void Start()
    {
        newRoomCanvas.enabled = false;
        mainCanvas.enabled = false;
        playerNameCanvas.enabled = false;
        connectingCanvas.enabled = true;
        Connect();

        if (PlayerPrefs.GetString("Name") == "")
        {
            PlayerPrefs.SetString("Name", "PLAYER");
        }

        playerNameDisplay.text = "Player Name: " + PlayerPrefs.GetString("Name");
    }


    #endregion


    #region Public Methods


    /// <summary>
    /// Start the connection process.
    /// - If already connected, we attempt joining a random room
    /// - if not yet connected, Connect this application instance to Photon Cloud Network
    /// </summary>
    public void Connect()
    {
        // we check if we are connected or not, we join if we are , else we initiate the connection to the server.
        if (PhotonNetwork.IsConnected)
        {
            // #Critical we need at this point to attempt joining a Random Room. If it fails, we'll get notified in OnJoinRandomFailed() and we'll create one.
            //PhotonNetwork.JoinRandomRoom();
        }
        else
        {
            // #Critical, we must first and foremost connect to Photon Online Server.
            PhotonNetwork.ConnectUsingSettings();
            PhotonNetwork.GameVersion = gameVersion;
        }
    }

    public void ListRooms()
    {
        //PhotonNetwork.Lis
    }

    public void OnClickCreateRoom()
    {
        Debug.Log("Create room clicked");
        mainCanvas.enabled = false;
        newRoomCanvas.enabled = true;
    }

    public void OnClickChangerPlayerName()
    {
        Debug.Log("Player name clicked");
        mainCanvas.enabled = false;
        playerNameCanvas.enabled = true;
    }

    public void OnClickBack()
    {
        Debug.Log("Cancel clicked");
        mainCanvas.enabled = true;
        newRoomCanvas.enabled = false;
        playerNameCanvas.enabled = false;

    }

    public void OnClickStart()
    {
        RoomOptions roomOptions = new RoomOptions() { IsVisible = true, IsOpen = true, MaxPlayers = 4, CleanupCacheOnLeave = true };

        if (PhotonNetwork.CreateRoom(buildPrompt.text, roomOptions, TypedLobby.Default))
        {
            Debug.Log("Created room " + buildPrompt.text);
            //PhotonNetwork.JoinRoom(buildPrompt.text);
        }
        else
        {
            Debug.Log("Create room failed!");
        }
    }

    public void OnClickSubmitPlayerName()
    {
        PlayerPrefs.SetString("Name", playerName.text);
        playerNameDisplay.text = "Player Name: " + playerName.text;

        OnClickBack();
    }


    #endregion


    #region MonoBehaviourPunCallbacks Callbacks


    public override void OnConnectedToMaster()
    {
        Debug.Log("PUN Basics Tutorial/Launcher: OnConnectedToMaster() was called by PUN");

        //PhotonNetwork.AutomaticallySyncScene = false;
        PhotonNetwork.NickName = PlayerPrefs.GetString("Name");

        PhotonNetwork.JoinLobby(TypedLobby.Default);

        connectingCanvas.enabled = false;
        mainCanvas.enabled = true;
    }


    public override void OnDisconnected(DisconnectCause cause)
    {
        Debug.LogWarningFormat("PUN Basics Tutorial/Launcher: OnDisconnected() was called by PUN with reason {0}", cause);

        connectingCanvas.enabled = true;
        mainCanvas.enabled = false;
        newRoomCanvas.enabled = false;
    }


    public override void OnJoinRandomFailed(short returnCode, string message)
    {
        Debug.Log("PUN Basics Tutorial/Launcher:OnJoinRandomFailed() was called by PUN. No random room available, so we create one.\nCalling: PhotonNetwork.CreateRoom");

        // #Critical: we failed to join a random room, maybe none exists or they are all full. No worries, we create a new room.
        PhotonNetwork.CreateRoom(null, new RoomOptions { MaxPlayers = 4 });
    }


    public override void OnJoinedRoom()
    {
        Debug.Log("PUN Basics Tutorial/Launcher: OnJoinedRoom() called by PUN. Now this client is in a room.");

        if (PhotonNetwork.IsMasterClient)
        {
            Debug.LogFormat("PhotonNetwork : Loading Level");
            PhotonNetwork.LoadLevel("BuildZone");
        }
    }

    public override void OnRoomListUpdate(List<RoomInfo> roomList)
    {
        Debug.Log("Room List Updated");

        // Destroy all existing buttons
        foreach(Transform child in joinButtonPanel.transform)
        {
            Destroy(child.gameObject);
        }

        foreach(RoomInfo room in roomList)
        {
            Debug.Log(room);
            if (room.IsVisible && room.PlayerCount < room.MaxPlayers)
            {
                GameObject roomButton = Instantiate(joinButtonPrefab);
                roomButton.transform.SetParent(joinButtonPanel.transform, false);
                roomButton.GetComponentInChildren<Text>().text = "Join: " + room.Name;
                roomButton.GetComponent<JoinRoomButton>().roomName = room.Name;
            }
        }
    }


    #endregion


}