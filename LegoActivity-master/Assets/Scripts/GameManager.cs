using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Photon.Pun;
using Photon.Realtime;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviourPunCallbacks, IPunInstantiateMagicCallback
{
    public GameObject playerPrefab;
    public GameObject LocalPlayerInstance;

    #region Photon Callbacks


    /// <summary>
    /// Called when the local player left the room. We need to load the launcher scene.
    /// </summary>
    public override void OnLeftRoom()
    {
        // Load the "Offline" Scene -- our Lobby
        SceneManager.LoadScene(0);
    }


    #endregion


    #region Public Methods

    // Start is called before the first frame update
    void Start()
    {
        if(PhotonNetwork.IsConnected)
        {
            PhotonNetwork.Instantiate(this.playerPrefab.name, new Vector3(50f, 10f, 50f), Quaternion.identity, 0);
        }
        else
        {
            GameObject player = Instantiate(playerPrefab);
            player.transform.position = new Vector3(50f, 10f, 50f);
        }
    }


    public void OnPhotonInstantiate(PhotonMessageInfo info)
    {
        //Debug.Log("On Photon int"); // seen when other disconnects

        if (PhotonNetwork.IsMasterClient)
        {
            //PhotonNetwork.Destroy(other.)
        }
    }

    //public override void OnPlayerLeftRoom(Player other)
    //{
    //    Debug.Log("OnPlayerLeftRoom() " + other.NickName); // seen when other disconnects

    //    if (PhotonNetwork.IsMasterClient)
    //    {
    //        PhotonNetwork.Destroy(other.)
    //    }
    //}

    public void LeaveRoom()
    {
        Debug.Log("Player is leaving the Room! Going back to Lobby...");
        PhotonNetwork.LeaveRoom();
    }

    #endregion
}
